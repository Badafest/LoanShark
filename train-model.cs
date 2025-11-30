#: package Microsoft.Data.Analysis@0.23.0
#: package Microsoft.ML@5.0.0

using Microsoft.Data.Analysis;
using Microsoft.ML;

const string DATASET_CSV_PATH = "./data.csv";
const string MISSING_VALUE_REPLACEMENT = "unknown";

const string MODEL_SAVE_PATH = "./model.zip";

DataFrame dataFrame = DataFrame.LoadCsv(DATASET_CSV_PATH);

// Rename the columns to lowercase
foreach (var column in dataFrame.Columns)
{
    dataFrame.Columns.RenameColumn(column.Name, column.Name.ToLower());
}

long numberOfRecords = dataFrame.Rows.Count;
// Handle the missing values
// Numerical columns -> Replace the null values with MODE (value with highest count)
// Categorical columns -> Replace the empty or null strings with "unknown" token
foreach (var column in dataFrame.Columns)
{
    bool isNumericColumn = column.IsNumericColumn();

    if (isNumericColumn)
    {
        // ValueCounts -> a dataframe with 2 columns: Values and Counts
        // This dataframe will have ALL DISTINCT VALUES with thier COUNTS
        var valueCounts = column.ValueCounts().OrderByDescending("Counts", putNullValuesLast: true);
        var mode = (float)valueCounts["Values"][0];
        column.FillNulls(mode, inPlace: true);
        continue;
    }

    int nullValues = 0;
    for (int i = 0; i < numberOfRecords; i++)
    {
        if (string.IsNullOrEmpty(dataFrame[column.Name][i]?.ToString()))
        {
            nullValues++;
            dataFrame[column.Name][i] = MISSING_VALUE_REPLACEMENT;
        }
    }
}

// Train the Binary Classifier Model -> Predict a "Label" based on "Features"
// Select relevant features
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
    dataFrame.Columns.Remove(column);
}

string labelColumnName = "status";

// Convert the numeric status column to boolean
// i.e, 1 to true and 0 to false
bool[] booleanStatuses = [.. (dataFrame[labelColumnName] as SingleDataFrameColumn)!.Select(status => status == 1)];
dataFrame.Columns.Remove(labelColumnName);
dataFrame.Columns.Add(new PrimitiveDataFrameColumn<bool>(labelColumnName, booleanStatuses));

// ACTUALLY TRAINING NOW!!!
MLContext mlContext = new();

// Preare the dataset (DONE)
// Train-test split (splitting the dataset into two separate dataset - training and testing)
// train on the training dataset
// evaluate predictions made on the testing dataset (see the metrics)

// Train test split
var trainTestData = mlContext.Data.TrainTestSplit(
    dataFrame,
    testFraction: 0.15// 15% of the total data is used for testing and 85% for training
);


DataFrame trainData = trainTestData.TrainSet.ToDataFrame(maxRows: numberOfRecords);
DataFrame testData = trainTestData.TestSet.ToDataFrame(maxRows: numberOfRecords);

string featuresColumnName = "features";
string normalizedFeaturesColumnName = "normalized_" + featuresColumnName;

DataFrameColumn[] featureColumns = [.. trainData.Columns.Where(column => column.Name != labelColumnName)];

// Pipeline -> series of chained methods to transform the dataframe into a trained model
IEstimator<ITransformer> dataProcessingPipeline =
// encode the categorical columns - one hot encoding

// gender - "male", "female" and "others"
// gender = "male"

// gender = [1,0,0]
mlContext.Transforms.Categorical.OneHotEncoding([
    ..featureColumns.Where(column => !column.IsNumericColumn())
    .Select(column => new InputOutputColumnPair(column.Name))
],
outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Indicator,
keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue
)
// features concatenation -> convert all features to a vector named "features"
// X -> y
.Append(mlContext.Transforms.Concatenate(featuresColumnName, [
    ..featureColumns.Select(column => column.Name)
]))
// relevant columns -> features and status
// normalization -> converting all columns to have mean = 0 and variance = 1
.Append(mlContext.Transforms.NormalizeMeanVariance(
    outputColumnName: normalizedFeaturesColumnName,
    inputColumnName: featuresColumnName
));

// Train the model
// classification, regression, recommendation, ...
var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
    labelColumnName: labelColumnName,
    featureColumnName: normalizedFeaturesColumnName
);

// Actually train the model fr
var model = dataProcessingPipeline.Append(trainer).Fit(trainData);

var testPredictions = model.Transform(trainData);

var metrics = mlContext.BinaryClassification.Evaluate(
    data: testPredictions,
    labelColumnName: labelColumnName
);

Console.WriteLine($"\n\nModel trained with accuracy: {metrics.Accuracy}\n\n");

// POSITIVE -> status = True
// NEGATIVE -> status = False
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

mlContext.Model.Save(model, (dataFrame as IDataView).Schema, MODEL_SAVE_PATH);
