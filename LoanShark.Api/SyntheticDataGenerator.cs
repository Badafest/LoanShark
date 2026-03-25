using System.Text.Json;
using System.Text.Json.Serialization;
using LoanShark.Api.Entities;

namespace LoanShark.Api;

public class SyntheticDataGenerator(string pathToCdfs)
{
    private readonly Dictionary<string, Dictionary<float, float>> _cdfs =
    JsonSerializer.Deserialize(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, pathToCdfs)),
        CdfsSourceGenerationContext.Default.DictionaryStringDictionarySingleSingle)!;

    private readonly Random _random = new();

    public LoanObservation[] Generate(ushort numObservations = 1)
    {
        return [..Enumerable.Range(0, numObservations)
        .Select(_ =>
            new LoanObservation()
            {
                // assumption: all the columns represent independent random variables
                LoanAmount = GetColumnValue(LoanObservation.LOAN_AMOUNT_NAME),
                RateOfInterest = GetColumnValue(LoanObservation.RATE_OF_INTEREST_NAME),
                UpfrontCharges = GetColumnValue(LoanObservation.UPFRONT_CHARGES_NAME),
                Term = GetColumnValue(LoanObservation.TERM_NAME),
                PropertyValue = GetColumnValue(LoanObservation.PROPERTY_VALUE_NAME),
                Income = GetColumnValue(LoanObservation.INCOME_NAME),
                CreditScore = GetColumnValue(LoanObservation.CREDIT_SCORE_NAME),
                Age = GetColumnValue(LoanObservation.AGE_NAME) switch
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
                },
                Region = GetColumnValue(LoanObservation.REGION_NAME) switch
                {
                    1f => "North-East",
                    2f => "North",
                    3f => "central",
                    4f => "south",
                    _ => "unknown"
                },
                DebtToIncomeRatio = GetColumnValue(LoanObservation.DTIR_NAME)
            }
        )];

    }

    private float GetColumnValue(string columnName)
    {
        // generate CDF value (random value between 0 & 1)
        var probability = _random.NextDouble();
        // find the "x" value that corresponds to the CDF value
        return _cdfs[columnName].First(kv => kv.Value > probability).Key;
    }
}

[JsonSourceGenerationOptions()]
[JsonSerializable(typeof(Dictionary<string, Dictionary<float, float>>))]
internal partial class CdfsSourceGenerationContext : JsonSerializerContext { }