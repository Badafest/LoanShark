using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using LoanShark.Ui;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.Services.AddScoped<GameData>();

builder.Services.AddScoped(_ => new Api(new Uri(builder.Configuration["ApiBaseUrl"]!)));

await builder.Build().RunAsync();
