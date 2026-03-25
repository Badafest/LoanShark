using Microsoft.ML.Data;

namespace LoanShark.Api.Entities;

public class LoanPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Status { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}