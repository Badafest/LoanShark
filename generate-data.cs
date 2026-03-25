#:package Microsoft.Data.Analysis@0.23.0
#:package Microsoft.ML@5.0.0
#:package Plotly.NET@5.0.0
#:package Plotly.NET.CSharp@0.13.0

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Plotly.NET.CSharp;

DataFrame dataFrame = DataFrame.LoadCsv("./processed-data.csv");

int totalDefaulted = dataFrame["status"].Cast<bool>()
    .Count(status => status);

Console.WriteLine($@"
Loan Default Rate: {Math.Round(100f * totalDefaulted / dataFrame.Rows.Count, 2)}%
");

// we don't want to generate data for status and ltv (loan_amount / property_value)
string[] skippedColumns = ["status", "ltv"];

foreach (var columnName in skippedColumns)
{
    dataFrame.Columns.Remove(columnName);
}


Console.WriteLine($"DATA SOURCE\n{dataFrame.Head(5)}\n");


// convert age and region columns to numeric for generation
// we should remember to convert back while making predictions

// see the distinct values for age and region
Console.WriteLine("UNIQUE AGE AND REGION");
Console.WriteLine(string.Join(", ", dataFrame["age"].Cast<string>().Distinct()));
Console.WriteLine(string.Join(", ", dataFrame["region"].Cast<string>().Distinct()));

// for age we map to their group averages
var floatAges = dataFrame["age"].Cast<string>()
    .Select(age => age switch
    {
        "<25" => 20f,
        "25-34" => 30f,
        "35-44" => 40f,
        "45-54" => 50f,
        "55-64" => 60f,
        "65-74" => 70f,
        ">74" => 80f,
        _ => 0f
    });

dataFrame["age"] = new SingleDataFrameColumn("", floatAges);

// for region we use arbitrary mapping
var floatRegions = dataFrame["region"].Cast<string>()
    .Select(region => region switch
    {
        "North-East" => 1f,
        "North" => 2f,
        "central" => 3f,
        "south" => 4f,
        _ => 0f
    });

dataFrame["region"] = new SingleDataFrameColumn("", floatRegions);

Console.WriteLine($"\nCONVERTED DATAFRAME\n{dataFrame.Head(10)}\n");

Chart.Grid(
    gCharts: dataFrame.Columns.Select(column => Chart.Histogram<float, string, string>(
        X: new(column.Cast<float>(), true),
        HistNorm: Plotly.NET.StyleParam.HistNorm.Percent
    )
    .WithXAxisStyle<float, string, string>(column.Name)),
    nRows: dataFrame.Columns.Count,
    nCols: 1,
    SubPlotTitles: new(dataFrame.Columns.Select(column => $"Histogram for {column.Name}"), true)
)
.WithSize(800, 500 * dataFrame.Columns.Count)
.SaveHtml($"./plots/histograms.html");


// cumulative distribution function

// x = [1,1,1,2,2,2,2,2,3,3,4,4,5,5,5,5]

// "valueCounts" data frame = frequency table
// index    x    f       cf (cumulative frequency)       probailities
// 0        1    3       3                                3/16
// 1        2    5       3 + 5 = 8                        1/2
// 2        3    2       8 + 2 = 10                       5/8
// 3        4    2       10 + 2 = 12                      3/4   
// 4        5    4       12 + 4 = 16                      1   

// x = Values, f = Counts

// cdf(4) = 75%
// The probability of the random variable "X" being less than or equal to 4 is 75% 

// "column_name" : {<x> : <probabilities>}

Dictionary<string, Dictionary<float, float>> cdfs = [];

List<Plotly.NET.GenericChart> cdfCharts = [];
foreach (var column in dataFrame.Columns)
{
    DataFrame valueCounts = column.ValueCounts().OrderBy("Values");

    long[] counts = [.. valueCounts["Counts"].Cast<long>()];
    long totalCount = counts.Sum();

    float[] probabilities = [
        ..Enumerable.Range(0, counts.Length)
        .Select(index => 1f * counts[0..(index + 1)].Sum() / totalCount)
    ];

    valueCounts.Columns.Add(new PrimitiveDataFrameColumn<float>(
        name: "Probabilities",
        probabilities
    ));

    // only keep 20 unique values per column
    int maxRows = 20;
    long numberOfRows = valueCounts.Rows.Count; // count of unique values

    // 5 -> [0, 1, 2, 3, 4] (step = 1 -> 5/20 + 1)
    // 28 -> [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27] (step = 2 -> 28/20 + 1)

    if (numberOfRows > maxRows)
    {
        var step = Math.Ceiling(1f * numberOfRows / maxRows);
        var numberOfIndices = (int)Math.Ceiling(1f * numberOfRows / step);

        long[] indices = [
            ..Enumerable.Range(0, numberOfIndices)
                .Select(index =>  index * (int) step)
        ];

        // make sure that the last index is included
        indices[^1] = numberOfRows - 1;

        valueCounts = valueCounts.Filter(new PrimitiveDataFrameColumn<long>(
            "", indices
        ));
    }

    float[] cdfValues = [.. valueCounts["Values"].Cast<float>()];
    float[] cdfProbabilities = [.. valueCounts["Probabilities"].Cast<float>()];

    // empirical CDF for the column
    Dictionary<float, float> cdf = cdfValues
        .Zip(cdfProbabilities, (value, probability) => new { value, probability })
        .ToDictionary(kv => kv.value, kv => kv.probability);

    cdfs.Add(column.Name, cdf);

    // visualize CDF
    cdfCharts.Add(Chart.Line<float, float, string>(x: cdfValues, y: cdfProbabilities)
        .WithXAxisStyle<float, float, string>(column.Name)
        .WithYAxisStyle<float, float, string>("Cumulative Probability"));
}
Chart.Grid(
    gCharts: cdfCharts,
    nRows: dataFrame.Columns.Count,
    nCols: 1,
    SubPlotTitles: new(dataFrame.Columns.Select(column => $"CDF for {column.Name}"), true)
)
.WithSize(800, 500 * dataFrame.Columns.Count)
.SaveHtml($"./plots/cdfs.html");

File.WriteAllText("./cdfs.json", JsonSerializer.Serialize(cdfs,
    CdfsSourceGenerationContext.Default.DictionaryStringDictionarySingleSingle));

var random = new Random();

cdfs = JsonSerializer.Deserialize(
    File.ReadAllText("./cdfs.json"),
    CdfsSourceGenerationContext.Default.DictionaryStringDictionarySingleSingle
)!;

int numObservations = 1000;

// assumption: all the columns represent independent random variables
var syntheticData = new DataFrame(cdfs.Select(kv => new SingleDataFrameColumn(
    // kv.Key -> column_name
    // kv.Value -> <x> : <probability>
    kv.Key, Enumerable.Range(0, numObservations).Select(_ =>
    {
        // generate CDF value (random value between 0 & 1)
        var probability = random.NextDouble();
        // find the "x" value that corresponds to the CDF value
        return kv.Value.First(kv => kv.Value > probability).Key;
    })
)));

// visually verify distribution
Chart.Grid(
    gCharts: syntheticData.Columns.Select(column => Chart.Histogram<float, string, string>(
        X: new(column.Cast<float>(), true),
        HistNorm: Plotly.NET.StyleParam.HistNorm.Percent
    )
    .WithXAxisStyle<float, string, string>(column.Name)),
    nRows: syntheticData.Columns.Count,
    nCols: 1,
    SubPlotTitles: new(syntheticData.Columns.Select(column => $"Histogram for {column.Name} (Synthetic Data)"), true)
)
.WithSize(800, 500 * syntheticData.Columns.Count)
.SaveHtml($"./plots/synthetic-histograms.html");


// revert age and region columns
var stringAges = syntheticData["age"].Cast<float>()
    .Select(age => age switch
    {
        0 => "unknown",
        < 25 => "<25",
        < 35 => "25-34",
        < 45 => "35-44",
        < 55 => "45-54",
        < 65 => "55-64",
        < 75 => "65-74",
        < 85 => ">74",
        _ => "unknown"
    });

syntheticData["age"] = new StringDataFrameColumn("", stringAges);

var stringRegions = syntheticData["region"].Cast<float>()
    .Select(region => region switch
    {
        1f => "North-East",
        2f => "North",
        3f => "central",
        4f => "south",
        _ => "unknown"
    });

syntheticData["region"] = new StringDataFrameColumn("", stringRegions);

// compute ltv
syntheticData["ltv"] = syntheticData["loan_amount"] / syntheticData["property_value"];

Console.WriteLine($"\nSYNTHETIC DATA\n{syntheticData.Head(5)}\n");


var mlContext = new MLContext();

var model = mlContext.Model.Load("./model.zip", out var inputSchema);

var gameplayData = model.Transform(syntheticData);

syntheticData["status"] = gameplayData.ToDataFrame(numObservations)["PredictedLabel"];

Console.WriteLine($"PREDICTIONS\n{syntheticData.Head(5)}\n");

int syntheticDefaulted = syntheticData["status"].Cast<bool>().Count(status => status);

Console.WriteLine($@"
Loan Default Rate: {Math.Round(100f * syntheticDefaulted / syntheticData.Rows.Count, 2)}%
");


[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(Dictionary<string, Dictionary<float, float>>))]
internal partial class CdfsSourceGenerationContext : JsonSerializerContext { }