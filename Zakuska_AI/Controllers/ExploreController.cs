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
                var Content = JsonConvert.SerializeObject(apiSendingData);
                var stringContent = new StringContent(Content, Encoding.UTF8, "application/json");
                Console.WriteLine(Content);
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
                            List<apiComingData> articleDatas = JsonConvert.DeserializeObject<List<apiComingData>>(responseString);

                            // Her bir öğe için article_name ve similarity değerlerini yazdırma
                            string[] sug = new string[articleDatas.Count];
                            string[] sim = new string[articleDatas.Count];

                            for (int i = 0; i < articleDatas.Count; i++)
                            {
                                var item = articleDatas[i];
                                //Console.WriteLine($"Article Name: {item.article_name}, Similarity: {item.similarity}");
                                sug[i] = item.article_name;
                                sim[i] = item.similarity;
                            }

                            

                            account.Suggestions = sug;
                            account.Similarities = sim;


                            mainAccount = new modelDto()
                            {
                                Name = user.Name,
                                SurName = user.SurName,
                                UserName = user.userName,
                                Interests = user.Interests.Split(','),
                                Suggestions = sug,
                        };

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
