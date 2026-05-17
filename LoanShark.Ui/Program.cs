using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using LoanShark.Ui;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped(_ => new Api(new Uri(builder.Configuration["ApiBaseUrl"]!)));
builder.Services.AddScoped<GameData>();

await builder.Build().RunAsync();
