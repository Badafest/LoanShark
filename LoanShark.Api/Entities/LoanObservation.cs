using Microsoft.ML.Data;

namespace LoanShark.Api.Entities;

public class LoanObservation
{
    [ColumnName(LOAN_AMOUNT_NAME)]
    public float LoanAmount { get; set; }

    [ColumnName(RATE_OF_INTEREST_NAME)]
    public float RateOfInterest { get; set; }

    [ColumnName(UPFRONT_CHARGES_NAME)]
    public float UpfrontCharges { get; set; }

    [ColumnName(TERM_NAME)]
    public float Term { get; set; }

    [ColumnName(PROPERTY_VALUE_NAME)]
    public float PropertyValue { get; set; }

    [ColumnName(INCOME_NAME)]
    public float Income { get; set; }

    [ColumnName(CREDIT_SCORE_NAME)]
    public float CreditScore { get; set; }

    [ColumnName(AGE_NAME)]
    public string Age { get; set; } = "";

    [ColumnName(REGION_NAME)]
    public string Region { get; set; } = "";

    [ColumnName(DTIR_NAME)]
    public float DebtToIncomeRatio { get; set; }

    [ColumnName(LTV_NAME)]
    public float LoanToValueRatio => PropertyValue == 0 ? 0 : LoanAmount / PropertyValue;

    // names of the fields
    public const string LOAN_AMOUNT_NAME = "loan_amount";
    public const string RATE_OF_INTEREST_NAME = "rate_of_interest";
    public const string UPFRONT_CHARGES_NAME = "upfront_charges";
    public const string TERM_NAME = "term";
    public const string PROPERTY_VALUE_NAME = "property_value";
    public const string INCOME_NAME = "income";
    public const string CREDIT_SCORE_NAME = "credit_score";
    public const string AGE_NAME = "age";
    public const string REGION_NAME = "region";
    public const string DTIR_NAME = "dtir1";
    public const string LTV_NAME = "ltv";
}