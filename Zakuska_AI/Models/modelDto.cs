namespace Zakuska_AI.Models
{
    public class modelDto
    {
        public string Name { get; set; }
        public string SurName { get; set; }

        public string UserName { get; set; }

        public string Email { get; set; }

        public string[] Interests { get; set; }

        public string[] Suggestions { get; set; }
        public string[] secondSuggestions { get; set; }

        public string[] Similarities { get; set; }
        public string[] secondSimilarities { get; set; }
    }
}
