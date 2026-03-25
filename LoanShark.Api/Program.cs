using System.ComponentModel.DataAnnotations;
using LoanShark.Api;
using LoanShark.Api.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

// Use prediction engine pool for thread safety and to reuse the prediction engines
// Reference: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/serve-model-web-api-ml-net
builder.Services
    .AddPredictionEnginePool<LoanObservation, LoanPrediction>()
    .FromFile(
        filePath: Path.Combine(AppContext.BaseDirectory, builder.Configuration["ModelPath"]!),
        watchForChanges: true);

// Synthetic data generator
builder.Services.AddSingleton(_ => new SyntheticDataGenerator(builder.Configuration["CdfsPath"]!));

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

// Validation
builder.Services.AddValidation();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

var group = app.MapGroup("/api");

group.MapGet("/generate", (
    SyntheticDataGenerator data,
    [FromQuery, Range(1, 100)] ushort n = 25) =>
        Results.Ok(data.Generate(n))
);

group.MapPost("/predict", async (
    PredictionEnginePool<LoanObservation, LoanPrediction> pool,
    [FromBody] LoanObservation[] observations
) =>
{
    var predictions = await Task.WhenAll(
        observations.Select(input =>
            Task.FromResult(pool.Predict(input))
    ));

    return Results.Ok(predictions);
}
);

app.Run();

