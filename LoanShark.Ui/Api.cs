
using System.Net.Http.Json;
using LoanShark.Ui.Entities;

namespace LoanShark.Ui;

public class Api(Uri baseAddress)
{
    private readonly HttpClient _httpClient = new()
    {
        BaseAddress = baseAddress
    };

    public async Task<ApplicantDetail[]> Generate(ushort n)
    {
        var applicants = await _httpClient.GetFromJsonAsync<ApplicantDetail[]>($"api/generate?n={n}");

        // validation of loan amount and upfront charges
        foreach (var applicant in applicants!)
        {
            applicant.LoanAmount = (float)Math.Clamp(applicant.LoanAmount, 0.2 * applicant.PropertyValue, 1.2 * applicant.PropertyValue);
            applicant.UpfrontCharges = (float)Math.Clamp(applicant.UpfrontCharges, 0, 0.1 * applicant.PropertyValue);
        }

        return applicants!;
    }

    public async Task<Prediction[]> Predict(ApplicantDetail[] applicants)
    {
        var response = await _httpClient.PostAsJsonAsync("api/predict", applicants);

        response.EnsureSuccessStatusCode();

        var predictions = await response.Content.ReadFromJsonAsync<Prediction[]>();
        return predictions!;
    }
}