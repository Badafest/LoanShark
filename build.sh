dotnet workload install wasm-tools

dotnet run train-model.cs
dotnet run generate-data.cs

dotnet publish -c Release -o ./Publish ./LoanShark.Ui/LoanShark.Ui.csproj
dotnet publish -c Release -o ./Publish ./LoanShark.Api/LoanShark.Api.csproj