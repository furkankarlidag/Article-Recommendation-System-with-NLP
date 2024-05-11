namespace Zakuska_AI.Models
{
    public class modelDto
    {
        public string Name { get; set; }
        public string SurName { get; set; }

        public string UserName { get; set; }

        public string Email { get; set; }

        public string[] Interests { get; set; }

        public List<apiComingData> Suggestions { get; set; }
        public List<apiComingData> SuggestionsScibert { get; set; }

    }
}
