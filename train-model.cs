#:package Microsoft.Data.Analysis@0.23.0
#:package Microsoft.ML@5.0.0

using System.Data;
using Microsoft.Data.Analysis;
using Microsoft.ML;

const string DATASET_CSV_PATH = "./data.csv";
const string MISSING_VALUE_REPLACEMENT = "unknown";

const string MODEL_SAVE_PATH = "./model.zip";

Console.Write($"STEP 1: Reading data from {DATASET_CSV_PATH}...");

// Import the CSV file as DataFrame
DataFrame dataFrame = DataFrame.LoadCsv(DATASET_CSV_PATH);

Console.WriteLine($"Done!\n\nSTEP 2: Data preprocessing...\n");

foreach (var column in dataFrame.Columns)
{
    // Rename columns to lowercase
    dataFrame.Columns.RenameColumn(column.Name, column.Name.ToLower());
}

// Drop columns irrelevant to the gameplay
string[] relevantColumns = [
    "age",
    "region",
    "income",
    "credit_score",
    "loan_amount",
    "upfront_charges",
    "property_value",
    "dtir1", // debt to income ratio
    "ltv", // loan amount to property value ratio
    "rate_of_interest",
    "term", // duration
    "status" // the label column for prediction
];

string[] irrelevantColumns = [.. dataFrame.Columns.Select(column => column.Name).Where(name => !relevantColumns.Contains(name))];

foreach (var column in irrelevantColumns)
{
    Console.WriteLine($"Dropping column {column}...");
    dataFrame.Columns.Remove(column);
}

// The `label` we are interested in is the "status" column.
string labelColumnName = "status";

// Convert the numeric status column to boolean
// i.e, 1 to true and 0 to false
bool[] booleanStatuses = [.. (dataFrame[labelColumnName] as SingleDataFrameColumn)!.Select(status => status == 1)];
dataFrame.Columns.Remove(labelColumnName);
dataFrame.Columns.Add(new PrimitiveDataFrameColumn<bool>(labelColumnName, booleanStatuses));

long numberOfRecords = dataFrame.Rows.Count;


foreach (var column in dataFrame.Columns)
{
    bool isNumericColumn = column.IsNumericColumn();

    // Value Counts - a dataframe with all unique values with counts for each column
    // It has two columns - "Values" and "Counts"
    // Take only "Top 10 values"
    DataFrame valueCounts = column.ValueCounts().OrderByDescending("Counts");


    // Fill the missing values in numeric columns with their MODE (item with highest freqeuncy)
    if (isNumericColumn)
    {
        var mode = (float)valueCounts["Values"][0];
        column.FillNulls(mode, inPlace: true);
        continue;
    }

    // Replace empty strings with MISSING_VALUE_REPLACEMENT for categorical columns
    for (int i = 0; i < numberOfRecords; i++)
    {
        if (string.IsNullOrEmpty(dataFrame[column.Name][i]?.ToString()))
        {
            dataFrame[column.Name][i] = MISSING_VALUE_REPLACEMENT;
        }
    }

}

Console.WriteLine("\n\nSTEP 3: Training the model...");

// Prepare the ML Context for feature selection, transformation and training
MLContext mlContext = new();

// Train test split
var trainTestData = mlContext.Data.TrainTestSplit(
    dataFrame,
    testFraction: 0.15// 15% of the total data is used for testing and 85% for training
);

DataFrame trainData = trainTestData.TrainSet.ToDataFrame(maxRows: dataFrame.Rows.Count);
DataFrame testData = trainTestData.TestSet.ToDataFrame(maxRows: dataFrame.Rows.Count);

// Preview train and test data
foreach (var dataset in new DataFrame[] { trainData, testData })
{
    var preview = dataset.Head(5);

    Console.WriteLine($"\n\nDataset Preview ({preview.Rows.Count}/{dataset.Rows.Count})\n");

    // Display column names
    foreach (var column in preview.Columns)
    {
        Console.Write($"{column.Name,-16}");
    }
    Console.WriteLine();

    // Display rows
    foreach (var row in preview.Rows)
    {
        foreach (var value in row)
        {
            Console.Write($"{value,-16}");
        }
        Console.WriteLine();
    }

}

// Name of features column name
const string featuresColumnName = "features";
const string normalizedFeaturesColumnName = "normalized_" + featuresColumnName;

DataFrameColumn[] featureColumns = [..trainData.Columns
    .Where(column => column.Name != labelColumnName)];

IEstimator<ITransformer> dataProcessingPipeline =
// encode categorical features - convert string values to numeric
mlContext.Transforms.Categorical.OneHotEncoding([
    ..featureColumns.Where(column => !column.IsNumericColumn())
        .Select(column => new InputOutputColumnPair(column.Name))
],
outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Indicator,
keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
// features concatenation - combine all relevant features to a single vector
.Append(mlContext.Transforms.Concatenate(featuresColumnName, [
    ..featureColumns.Select(column => column.Name)
]))
// normalize features - transform for mean = 0 and variance = 1
.Append(mlContext.Transforms.NormalizeMeanVariance(
    outputColumnName: normalizedFeaturesColumnName,
    inputColumnName: featuresColumnName
));

// Train the model
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
    labelColumnName: labelColumnName,
    featureColumnName: normalizedFeaturesColumnName
);

var model = dataProcessingPipeline.Append(trainer).Fit(trainData);

// Evaluate the model
var testPredictions = model.Transform(testData);

var metrics = mlContext.BinaryClassification.Evaluate(
    data: testPredictions,
    labelColumnName: labelColumnName
);

// Display the model metrics
Console.WriteLine($"\n\nModel trained with accuracy: {metrics.Accuracy}\n\n");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

Console.Write($"\nSaving model: {MODEL_SAVE_PATH}...");
// Save the model for future use
mlContext.Model.Save(model, (dataFrame as IDataView).Schema, MODEL_SAVE_PATH);
Console.Write("Done!");