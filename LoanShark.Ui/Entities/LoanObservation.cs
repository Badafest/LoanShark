namespace LoanShark.Ui.Entities;

public class LoanObservation
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
}