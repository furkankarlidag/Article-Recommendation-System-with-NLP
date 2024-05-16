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
using System.Globalization;

namespace Zakuska_AI.Controllers
{
    public class ExploreController : Controller
    {
        private readonly UserManager<AppUser> _userManager;
        private readonly SignInManager<AppUser> _signInManager;
        stringSQL strSQL = new stringSQL();
        public static modelDto mainAccount;
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
                mainAccount = new modelDto();
                mainAccount = account;


                return View(account);
            }
            return View();
        }

        

        [HttpPost]
        public async Task<IActionResult> FeedBack(string action, string articleKeywords,string userId)
        {
            string baseUrl = "http://127.0.0.1:8000/";
            string[] keywords = articleKeywords.Split(',');

            if (action == "like")
            {
                var Content = JsonConvert.SerializeObject(keywords);
                var stringContent = new StringContent(Content, Encoding.UTF8, "application/json");
                
                
                string endpoint = "like/";
                

              
                string apiURL = $"{baseUrl}{endpoint}{userId}";
                using (HttpClient client = new HttpClient())
                {
                    Console.WriteLine(apiURL + "-----");
                    client.Timeout = Timeout.InfiniteTimeSpan;
                    Console.WriteLine(Content);
                    try
                    {
                        var res = await client.PostAsync(apiURL, stringContent);
                        if (res.IsSuccessStatusCode)
                        {
                            Console.WriteLine("like succesfull " + stringContent + userId);
                        }
                        
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured here amk: " + e);
                    }
                }
                return NoContent();
            }
            else if (action == "dislike")
            {
                var Content = JsonConvert.SerializeObject(keywords);
                var stringContent = new StringContent(Content, Encoding.UTF8, "application/json");


                string endpoint = "dislike/";



                string apiURL = $"{baseUrl}{endpoint}{userId}";
                using (HttpClient client = new HttpClient())
                {
                    Console.WriteLine(apiURL + "-----");
                    client.Timeout = Timeout.InfiniteTimeSpan;
                    Console.WriteLine(Content);
                    try
                    {
                        var res = await client.PostAsync(apiURL, stringContent);
                        if (res.IsSuccessStatusCode)
                        {
                            Console.WriteLine("dislike succesfull " + stringContent + userId);
                        }

                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("Error occured here amk: " + e);
                    }
                }
                return NoContent();
            }

            return NoContent();
        }

        [HttpPost]
        public async Task<IActionResult> Search(string searchQuery,string userName)
        {
            string baseUrl = "http://127.0.0.1:8000/"; 
            string endPoint = "search_fasttext/";
            string secondEndpoint = "search_scibert/";

            List<apiComingData> veriler = new List<apiComingData> { };
            string apiUrl = $"{baseUrl}{endPoint}{userName}?searchKey={searchQuery}";
            
            using (HttpClient client = new HttpClient())
            {
                client.Timeout = Timeout.InfiniteTimeSpan;
                try
                {
                    var res = await client.PostAsync(apiUrl,null);
                    if (res.IsSuccessStatusCode)
                    {
                        string responseString = await res.Content.ReadAsStringAsync();
                        //Console.WriteLine(responseString);
                        List<apiComingData> datas = JsonConvert.DeserializeObject<List<apiComingData>>(responseString);

                        mainAccount.FasttextSearchResults = datas;
                        //RedirectToAction("SearchDetails", mainAccount);
                        
                        
                    }
                    
                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured here amk: " + e);
                }
            }

            string apiUrl2 = $"{baseUrl}{secondEndpoint}{userName}?searchKey={searchQuery}";
            using (HttpClient client = new HttpClient())
            {
                client.Timeout = Timeout.InfiniteTimeSpan;
                try
                {
                    var res = await client.PostAsync(apiUrl2, null);
                    if (res.IsSuccessStatusCode)
                    {
                        string responseString = await res.Content.ReadAsStringAsync();
                        //Console.WriteLine(responseString);
                        List<apiComingData> datas = JsonConvert.DeserializeObject<List<apiComingData>>(responseString);

                        mainAccount.ScibertSearchResults = datas;
                        //RedirectToAction("SearchDetails", mainAccount);


                    }

                }
                catch (Exception e)
                {
                    Console.WriteLine("Error occured here amk: " + e);
                }
            }
            return RedirectToAction("SearchDetails", mainAccount);
        }
        [HttpGet]
        public IActionResult SearchDetails(modelDto data)
        {
           
            return View("Index", mainAccount);
        }
        public IActionResult Detay (string str)
        {
            apiComingData data = new apiComingData();
            for (int i = 0; i < 5; i++)
            {
                if(str == mainAccount.Suggestions[i].name)
                {
                    data = mainAccount.Suggestions[i];
                    
                    return View("Detay", data);
                }
            }
            for (int i = 0; i < 5; i++)
            {
                if (str == mainAccount.SuggestionsScibert[i].name)
                {
                    data = mainAccount.SuggestionsScibert[i];
                    return View("Detay", data);
                }
            }
            for (int i = 0; i < 5; i++)
            {
                if (str == mainAccount.FasttextSearchResults[i].name)
                {
                    data = mainAccount.FasttextSearchResults[i];
                    return View("Detay", data);
                }
            }
            for (int i = 0; i < 5; i++)
            {
                if (str == mainAccount.ScibertSearchResults[i].name)
                {
                    data = mainAccount.ScibertSearchResults[i];
                    return View("Detay", data);
                }
            }


            return NoContent();
        }
    }
}
