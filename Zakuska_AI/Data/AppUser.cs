using Microsoft.AspNetCore.Identity;

namespace Zakuska_AI.Data
{
    public class AppUser : IdentityUser<int>
    {
        public string Name { get; set; }
        public string SurName { get; set; }
        public string Interests { get; set; }

    }
}
