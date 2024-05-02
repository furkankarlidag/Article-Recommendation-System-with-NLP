using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Zakuska_AI.Data;
using Zakuska_AI.Models;

namespace Zakuska_AI.Controllers
{
    public class AccountController : Controller
    {
        private  UserManager<AppUser> _userManager;
        private readonly SignInManager<AppUser> _signInManager;

       
        public AccountController(UserManager<AppUser> userManager, SignInManager<AppUser> signInManager)
        {
            _userManager = userManager;
            _signInManager = signInManager;
        }

        public IActionResult SignIn()
        {
            return View();
        }


        [HttpPost]
        public async Task<IActionResult> SignIn(OldUser user)
        {
            if (!ModelState.IsValid)
            {
                return View(user);
            }
            else
            {
                var userMail = await _userManager.FindByEmailAsync(user.Email);

                if (userMail == null)
                {
                    ModelState.AddModelError("", "Bu Email ile daha önce hesap oluşturulmamış!!");
                    return View(user);
                }
                var result = await _signInManager.PasswordSignInAsync(userMail, user.Password, true, true);
                if (result.Succeeded)
                {
                    TempData["UserName"] = user.Email;
                    return RedirectToAction("Index", "Explore");
                }
                else
                {
                    ModelState.AddModelError("", "Girilen email veya parola yanlış");
                    return View(user);
                }
                
            }


            return RedirectToAction("Index", "Explore");

        }
        public IActionResult SignUp()
        {

            return View();
        }

        [HttpPost]
        public async Task<IActionResult> SignUp(NewUser user)
        {
            string selectedInterests = string.Empty;
            if (!ModelState.IsValid)
            {
                return View(user);
            }
            else
            {
                bool interestsSelected = user.Interests != null && user.Interests.Length > 0;

                if (interestsSelected)
                {
                    selectedInterests = string.Join(", ", user.Interests);
                }

                AppUser appUser = new AppUser()
                {
                    Name = user.Name,
                    SurName = user.Surname,
                    Email = user.Email,
                    UserName = user.UserName,

                    Interests = selectedInterests,
                };
                var result = await _userManager.CreateAsync(appUser,user.Password);
                if (result.Succeeded)
                {
                    TempData["SuccessMessage"] = "Kayıt işlemi başarıyla tamamlandı.";
                    return RedirectToAction("SignIn", "Account");
                }
                else
                {
                    foreach(var item in result.Errors)
                    {
                        ModelState.AddModelError("", item.Description);
                    }
                }
                return View(user);
            }
        }


        [HttpPost]
        public async Task<IActionResult> Logout()
        {
            await _signInManager.SignOutAsync();
            return RedirectToAction("Index", "Home");
        }

    }
}
