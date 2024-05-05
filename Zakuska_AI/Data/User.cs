using Microsoft.EntityFrameworkCore;

namespace Zakuska_AI.Data
{
    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string SurName { get; set; }
        public string userName { get; set; }
        public string Email { get; set; }
        public string Interests { get; set; }
        public string SearchHistory { get; set; }


    }
}
