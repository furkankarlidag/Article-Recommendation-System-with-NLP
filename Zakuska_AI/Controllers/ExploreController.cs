using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Zakuska_AI.Data;
using Zakuska_AI.Models;

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
                NewUser account = new NewUser()
                {
                    Name = user.Name,
                    Surname = user.SurName,
                   UserName = user.userName,
                    Interests = user.Interests.Split(','),
                };
                return View(account);
            }
            return View();
        }

    }
}
