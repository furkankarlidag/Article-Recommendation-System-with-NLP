using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Zakuska_AI.Data
{
    public class Context : IdentityDbContext<AppUser,AppRole,int>
    {
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlServer("Server=GIGABYTE-LAPTOP;Database=yazlab3;Trusted_Connection = true;TrustServerCertificate=true;");
        }
        public DbSet<User> Users { get; set; }
    }
}
