FROM mcr.microsoft.com/dotnet/aspnet:10.0

WORKDIR /app
COPY Publish/ .

ENTRYPOINT ["dotnet", "LoanShark.API.dll"]
