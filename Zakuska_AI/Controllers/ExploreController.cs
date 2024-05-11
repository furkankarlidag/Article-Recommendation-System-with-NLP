﻿using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Zakuska_AI.Data;
using Zakuska_AI.Models;
using System.Net.Http;
using Azure;
using Azure.Core;
using Newtonsoft.Json;
using System.Text;
using System.Globalization;

namespace Zakuska_AI.Controllers
{
    public class ExploreController : Controller
    {
        private readonly UserManager<AppUser> _userManager;
        private readonly SignInManager<AppUser> _signInManager;
        stringSQL strSQL = new stringSQL();
        modelDto mainAccount;
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
                ApiSendingData apiSendingData = new ApiSendingData();
                apiSendingData.Interests = account.Interests;
                apiSendingData.UserName = account.UserName;
                var Content = JsonConvert.SerializeObject(apiSendingData.Interests);
                var stringContent = new StringContent(Content, Encoding.UTF8, "application/json");
                Console.WriteLine(Content);
                string baseUrl = "http://127.0.0.1:8000/";
                string endpoint = "recommendation/";
                string userId = apiSendingData.UserName;

                // Parametreleri birleştirerek tam URL oluştur
                string apiURL = $"{baseUrl}{endpoint}{userId}";
                using (HttpClient client = new HttpClient())
                {
                    client.Timeout = Timeout.InfiniteTimeSpan;

                    try
                    {
                        var res = await client.PostAsync(apiURL,stringContent);
                        if (res.IsSuccessStatusCode)
                        {
                            string responseString = await res.Content.ReadAsStringAsync();
                            Console.WriteLine(responseString);
                            List<apiComingData> datas = JsonConvert.DeserializeObject<List<apiComingData>>(responseString);
                            
                            List<apiComingData> veriler = new List<apiComingData>();
                            int i = 0;
                            account.Suggestions = datas;
                        }

                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured here amk: " + e);
                    }
                }
                string endpoint2 = "recommendation_scibert/";
                string api2URL = $"{baseUrl}{endpoint2}{userId}";
                using (HttpClient client = new HttpClient())
                {
                    client.Timeout = Timeout.InfiniteTimeSpan;

                    try
                    {
                        var res = await client.PostAsync(api2URL, stringContent);
                        if (res.IsSuccessStatusCode)
                        {
                            string responseString = await res.Content.ReadAsStringAsync();
                            Console.WriteLine(responseString);
                            List<apiComingData> datas = JsonConvert.DeserializeObject<List<apiComingData>>(responseString);

                            List<apiComingData> veriler = new List<apiComingData>();
                            int i = 0;
                            account.SuggestionsScibert = datas;
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
        [HttpPost]
        public IActionResult FeedBack(string action, string makaleId)
        {
            
            // action parametresine göre gerekli işlemler yapılabilir
            if (action == "like")
            {
                Console.WriteLine($"Feed back liked  {makaleId}");
                return NoContent();
            }
            else if (action == "dislike")
            {
                Console.WriteLine($"Feed back unliked  {makaleId}");
                return NoContent();
            }

            return NoContent();
        }

    }
}
