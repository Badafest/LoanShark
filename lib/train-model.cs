#:package Microsoft.Data.Analysis@0.23.0
#:package Microsoft.ML@5.0.0

using Microsoft.Data.Analysis;
using Microsoft.ML;

const string RAW_DATA_PATH = "../data/data.csv";
const string TRAIN_DATA_PATH = "../data/train-data.csv";
const string TEST_DATA_PATH = "../data/test-data.csv";
const string PROCESSED_DATA_PATH = "../data/processed-data.csv";
const string MODEL_PATH = "./model.zip";

// dataFrame -> tabular data (columns and rows)
DataFrame dataFrame = DataFrame.LoadCsv(RAW_DATA_PATH);

// Select relevant features
string[] relevantColumns = [
    // FEATURES
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

    // LABEL
    "status" // the label column for prediction
];

string[] columnNames = [.. dataFrame.Columns.Select(column => column.Name)];

foreach (var originalColumnName in columnNames)
{
    string columnName = originalColumnName.ToLower();

    // Rename columns to lowercase
    dataFrame.Columns.RenameColumn(originalColumnName, columnName);

    // Drop columns irrelevant to the gameplay
    if (!relevantColumns.Contains(columnName))
    {
        dataFrame.Columns.Remove(columnName: columnName);
        continue;
    }

    // Handling the missing values
    // Two types of columns -> Numeric (float) and Categorical (string)
    // Numeric Column -> Replace NULLs with MODE (value with highest count)
    // Categorical Column -> Replace NULLs and ""s with "unknown"
    var column = dataFrame.Columns[columnName];

    bool isNumericColumn = column.IsNumericColumn();

    // numeric column
    if (isNumericColumn)
    {
        // valueCounts is a DataFrame with two columns -> "Values" and "Counts"
        DataFrame valueCounts = column.ValueCounts();
        var mode = (float)valueCounts.OrderByDescending("Counts")["Values"][0];
        column.FillNulls(mode, inPlace: true);
        continue;
    }

    // categorical column
    dataFrame[columnName] = new StringDataFrameColumn(columnName, [
        ..(dataFrame[columnName] as StringDataFrameColumn)!.Select(value =>
            string.IsNullOrEmpty(value)?"unknown":value)
    ]);
}

string labelColumnName = "status";

// convert the label column to boolean
dataFrame[labelColumnName] = new PrimitiveDataFrameColumn<bool>(labelColumnName,
    [.. (dataFrame[labelColumnName] as PrimitiveDataFrameColumn<float>)!
        .Select(status => status == 1)]
);

// Save the processed data
DataFrame.SaveCsv(dataFrame, PROCESSED_DATA_PATH);

// Train the Model (Binary Classifier)

// features (X) -> all data used as "input" for the model
// label (y) -> the "output" (prediction) of the model
// Binary classifier -> the "label" can have 2 possible values (status = 0 or 1)

MLContext mlContext = new();

string featuresColumnName = "features";
string normalizedFeaturesColumnName = "normalized_" + featuresColumnName;

DataFrameColumn[] featuresColumns = [.. dataFrame.Columns
    .Where(column => column.Name != labelColumnName)];


// split the dataframe to two subsets - train data and test data -> Train-Test Split
// 0.15 -> 15% of the data is used for testing, 85% for training
var trainTestData = mlContext.Data.TrainTestSplit(dataFrame, testFraction: 0.15);

var trainData = trainTestData.TrainSet.ToDataFrame(maxRows: dataFrame.Rows.Count);
var testData = trainTestData.TestSet.ToDataFrame(maxRows: dataFrame.Rows.Count);

DataFrame.SaveCsv(trainData, TRAIN_DATA_PATH);
DataFrame.SaveCsv(testData, TEST_DATA_PATH);

// untrained transformer -> pipeline with steps to "transform" the data to weights of the final model
// 1. Encode the categorical columns (everything in the model has to be numeric)
var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
    [.. featuresColumns
        .Where(column => !column.IsNumericColumn())
        .Select(column => new InputOutputColumnPair(column.Name))]
)

// 2. Feature concatenation (we use the "X" matrix as input)
.Append(mlContext.Transforms.Concatenate(
    outputColumnName: featuresColumnName,
    inputColumnNames: [.. featuresColumns.Select(column => column.Name)])

// 3. Normalization (mean = 0 & variance = 1 -> improve speed and stability)
).Append(mlContext.Transforms.NormalizeMeanVariance(
    outputColumnName: normalizedFeaturesColumnName,
    inputColumnName: featuresColumnName)

// 4. Train the classifier
).Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
    labelColumnName: labelColumnName,
    featureColumnName: normalizedFeaturesColumnName
));

// To train the model (Trained Transformer) -> <UNTRAINED_TRANSFORMER>.Fit(training_data: IDataView) 
// (can be saved for future use)
var model = dataProcessingPipeline.Fit(trainData);

mlContext.Model.Save(model, (trainData as IDataView).Schema, MODEL_PATH);

// To make predictions (actual transformations) -> <TRAINED_TRANSFORMER>.Transform(actual_data: IDataView)
var predictions = model.Transform(testData);

var metrics = mlContext.BinaryClassification.Evaluate(
    data: predictions,
    labelColumnName: labelColumnName
);

Console.WriteLine($"Accuracy of the model: {metrics.Accuracy}");

Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
