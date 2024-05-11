namespace Zakuska_AI.Models
{
    public class apiComingData
    {
        public string _id { get; set; }
        public string name { get; set; }
        public string title { get; set; }
        public string abstractText { get; set; }
        public string fulltext { get; set; }
        public List<string> keywords { get; set; }
        public double similarity { get; set; }
    }
}
