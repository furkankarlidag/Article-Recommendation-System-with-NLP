using System.ComponentModel.DataAnnotations;

namespace Zakuska_AI.Models
{
    public class NewUser
    {
        [Required]
        public string Name { get; set; }

        [Required]
        public string Surname { get; set; }

        [DataType(DataType.EmailAddress)]
        [Required]
        public string Email { get; set; }
        [Required]
        public string  UserName { get; set; }
        [DataType(DataType.Password)]
        [Required]
        public string Password { get; set; }

        [DataType(DataType.Password)]
        [Required]
        [Compare("Password", ErrorMessage = "Sifreler birbiri ile eslesmiyor!!")]
        public string ConfirmPassword { get; set; }

        [Required]
        public string[] Interests { get; set; }
    }
}
