using System.ComponentModel.DataAnnotations;

namespace Zakuska_AI.Models
{
    public class OldUser
    {
        [Required]
        [DataType(DataType.EmailAddress)]
        public string Email { get; set; }
        [Required]
        [DataType(DataType.Password)]
        public string Password { get; set; }
    }
}
