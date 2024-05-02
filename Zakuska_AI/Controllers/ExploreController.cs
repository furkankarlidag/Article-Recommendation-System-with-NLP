using Microsoft.AspNetCore.Mvc;
using Zakuska_AI.Models;

namespace Zakuska_AI.Controllers
{
    public class ExploreController : Controller
    {
        public IActionResult Index()
        {
            var userName = TempData["UserName"] as string;
            NewUser us = new NewUser();
            us.UserName = userName;
            return View(us);
        }
    }
}
