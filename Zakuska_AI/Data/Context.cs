using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using Zakuska_AI.Models;

namespace Zakuska_AI.Data
{
    public class Context : IdentityDbContext<AppUser,AppRole,int>
    {
        stringSQL strSQL = new stringSQL();
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlServer(strSQL.SQLString);
        }
        public Context(DbContextOptions<Context> options) : base(options)
        {

        }
        public DbSet<User> Users { get; set; }
    }
}
