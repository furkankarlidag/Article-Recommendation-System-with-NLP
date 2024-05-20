import nltk
import spacy
import pandas as pd  
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datasets import load_dataset
from datasets import Dataset
import fasttext.util
from scipy.spatial.distance import cosine
import numpy as np



dataset = load_dataset("memray/krapivin", "default")



# sample from the train split
#print("Sample from training dataset split")
#train_sample = dataset["train"][0]
#print("Fields in the sample: ", [key for key in train_sample.keys()])
#print("Tokenized Document: ", train_sample["document"])
#print("Document BIO Tags: ", train_sample["doc_bio_tags"])
#print("Extractive/present Keyphrases: ", train_sample["extractive_keyphrases"])
#print("Abstractive/absent Keyphrases: ", train_sample["abstractive_keyphrases"])
#print("\n-----------\n")

# sample from the validation split
#print("Sample from validation dataset split")
#validation_sample = dataset["validation"][0]
#print("Fields in the sample: ", [key for key in validation_sample.keys()])
#print("Tokenized Document: ", validation_sample["document"])
#print("Document BIO Tags: ", validation_sample["doc_bio_tags"])
#print("Extractive/present Keyphrases: ", validation_sample["extractive_keyphrases"])
#print("Abstractive/absent Keyphrases: ", validation_sample["abstractive_keyphrases"])
#print("\n-----------\n")

# sample from the test split
#print("Sample from test dataset split")
#test_sample = dataset["test"][0]
#print("Fields in the sample: ", [key for key in test_sample.keys()])
#print("Tokenized Document: ", test_sample["document"])
#print("Document BIO Tags: ", test_sample["doc_bio_tags"])
#print("Extractive/present Keyphrases: ", test_sample["extractive_keyphrases"])
#print("Abstractive/absent Keyphrases: ", test_sample["abstractive_keyphrases"])
#print("\n-----------\n")


nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stop_words = set(stopwords.words('english'))

fasttext.util.download_model('en', if_exists='ignore') 
ft_model = fasttext.load_model('cc.en.300.bin')


def preprocess_text(text):
    doc = nlp(text.lower())  
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]

    return ' '.join(lemmas)



sample_text = "Machine learning, a subset of artificial intelligence, has rapidly evolved in recent years, transforming various industries and revolutionizing how we interact with technology."
processed_text = preprocess_text(sample_text)

print(processed_text)

ilgiAlanim = [
    "HTTP Protocol",
    "Web Server Software",
    "Client-Server Communication",
    "Server Management",
    "Web Security",
    "SSL/TLS",
    "HTTPS",
    "Server Load Balancing",
    "Server Monitoring and Maintenance",
    "Server Backup and Recovery",
    "Performance Optimization",
    "Database Integration",
    "API Presentation",
    "Web Services",
    "Server Configuration",
    "Access Control",
    "Server Logging and Monitoring",
    "Intrusion Detection and Prevention",
    "Server Configuration Management",
    "Server Update and Upgrade",
    
]



def get_fasttext_vector(text):
    clean_text = ' '.join(text.replace('\n', ' ').split())
    return ft_model.get_sentence_vector(clean_text)

print(ilgiAlanim)

ilgiAlanim_vektörleri = {}
index_ofseti = 0

for ilgi in ilgiAlanim:
    ilgi_metni = preprocess_text(ilgi)
    print(ilgi_metni)  
    if ilgi_metni:  
        ilgiAlanim_vektörleri[index_ofseti] = get_fasttext_vector(ilgi_metni)
        index_ofseti += 1

processed_text = get_fasttext_vector(processed_text)

article_vectors = {}
article_titles = {}
article_id = 0
index_offset = 0  
for split in ['validation', 'test']:
    for article in dataset[split]:
        article_text = " ".join(article["abstract"])
        article_title = article['title']
        #article_text = preprocess_text(article_text) 
        if article_text:  
            article_vectors[index_offset] = get_fasttext_vector(article_text)
            article_titles[article_id] = article_title 
            index_offset += 1
            article_id += 1


def calculate_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

# Tüm ilgi alanı vektörlerinin ortalamasını hesapla
def get_average_interest_vector(interest_vectors):    
    all_vectors = np.array(list(interest_vectors.values()))
    average_vector = np.mean(all_vectors, axis=0)
    return average_vector

def recommend_articles_based_on_interests(interest_vectors, article_vectors):
    average_interest_vector = get_average_interest_vector(interest_vectors)
    similarities = []

    for article_id, article_vector in article_vectors.items():
        similarity = calculate_similarity(average_interest_vector, article_vector)
        similarities.append((article_titles[article_id], similarity))  # ID yerine başlık kullan

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = similarities[:5]
    return top_recommendations

recommendations = recommend_articles_based_on_interests(ilgiAlanim_vektörleri, article_vectors)
print("Recommended articles:")
for rec in recommendations:
    print(f"Article Name: {rec[0]:<125} Similarity: {rec[1]:.5f}")

def calculate_precision_recall(recommended_articles, ground_truth_data):
    tp = sum(1 for rec in recommended_articles if rec in ground_truth_data)
    fp = sum(1 for rec in recommended_articles if rec not in ground_truth_data)
    fn = sum(1 for true in ground_truth_data if true not in recommended_articles)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

# Örnek kullanım
ground_truth_data = set([article['title'] for article in dataset['validation'] if 'desired_keyword' in article['keywords']])
recommended_titles = set([rec[0] for rec in recommendations])

precision, recall = calculate_precision_recall(recommended_titles, ground_truth_data)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")


