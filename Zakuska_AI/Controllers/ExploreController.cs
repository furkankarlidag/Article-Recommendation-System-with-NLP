using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Zakuska_AI.Data;
using Zakuska_AI.Models;
using System.Net.Http;
using Azure;
using Azure.Core;
using Newtonsoft.Json;
using System.Text;

namespace Zakuska_AI.Controllers
{
    public class ExploreController : Controller
    {
        private readonly UserManager<AppUser> _userManager;
        private readonly SignInManager<AppUser> _signInManager;
        stringSQL strSQL = new stringSQL();
        public ExploreController(UserManager<AppUser> userManager, SignInManager<AppUser> signInManager)
        {
            _userManager = userManager;
            _signInManager = signInManager;
        }

        public async Task<IActionResult> Index()
        {
            var userName = TempData["UserName"] as string;

            var optionsBuilder = new DbContextOptionsBuilder<Context>();
            optionsBuilder.UseSqlServer(strSQL.SQLString);
            var context = new Context(optionsBuilder.Options);
            var user  = context.Users.FirstOrDefault(x => x.userName == userName);
     
            if (user != null)
            {
                modelDto account = new modelDto()
                {
                    Name = user.Name,
                    SurName = user.SurName,
                    UserName = user.userName,
                    Interests = user.Interests.Split(','),
                };
                var Content = JsonConvert.SerializeObject(account.Interests);
                var stringContent = new StringContent(Content, Encoding.UTF8, "application/json");
                string apiURL = "http://127.0.0.1:5000/api/process_data";
                using(HttpClient client = new HttpClient())
                {
                    client.Timeout = Timeout.InfiniteTimeSpan;

                    try
                    {
                        var res = await client.PostAsync(apiURL,stringContent);
                        if (res.IsSuccessStatusCode)
                        {
                            string responseString = await res.Content.ReadAsStringAsync();
                            account.Suggestions = JsonConvert.DeserializeObject<string[]>(responseString);

                        }

                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured here amk: " + e);
                    }
                }
                return View(account);
            }
            return View();
        }

    }
}
