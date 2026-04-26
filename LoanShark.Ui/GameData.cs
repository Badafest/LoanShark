using LoanShark.Ui.Entities;

namespace LoanShark.Ui;

public enum GameState
{
    WAITING,
    APPLICANTS_LOADING,
    APPLICANTS_LOADED,
    APPLICANTS_FAILED,
    PREDICTING,
    PREDICTION_SUCCESS,
    PREDICTION_FAILED
}

public class GameData(Api api)
{
    private readonly Api _api = api;
    public GameState State { get; private set; }
    public readonly ushort TotalApplicants = 10;
    public float Balance = 1000000;
    public float LentAmount = 0;
    public ApplicantDetail[] Applicants { get; private set; } = [];
    public Prediction[] Predictions { get; set; } = [];

    public async Task LoadApplicantDetails()
    {
        State = GameState.APPLICANTS_LOADING;
        try
        {
            Applicants = await _api.Generate(TotalApplicants);
            State = GameState.APPLICANTS_LOADED;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"APPLICANTS FAILED: {ex}");
            State = GameState.APPLICANTS_FAILED;
        }
    }

    public async Task MakePredictions()
    {
        State = GameState.PREDICTING;
        try
        {
            Predictions = await _api.Predict(Applicants);
            State = GameState.PREDICTION_SUCCESS;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"PREDICTION FAILED: {ex}");
            State = GameState.PREDICTION_FAILED;
        }
    }

    public void Reset()
    {
        Applicants = [];
        Predictions = [];
        State = GameState.WAITING;
    }
}