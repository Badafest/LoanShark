namespace LoanShark.Ui.Entities;

public class ApplicantDetail
{
    public float LoanAmount { get; set; }
    public float RateOfInterest { get; set; }
    public float UpfrontCharges { get; set; }
    public float Term { get; set; }
    public float PropertyValue { get; set; }
    public float Income { get; set; }
    public float CreditScore { get; set; }
    public string Age { get; set; } = "";
    public string Region { get; set; } = "";
    public float DebtToIncomeRatio { get; set; }
    public bool Status { get; set; }
    public float EarnableInterest => LoanAmount * (float)Math.Pow(1 + RateOfInterest / 100, Term / 12);
    public float SalvageValue => PropertyValue * (float)Math.Pow(1 + 0.1 * RateOfInterest / 100, Term / 12);
    public float EarnedProfit { get; set; } = 0;
}