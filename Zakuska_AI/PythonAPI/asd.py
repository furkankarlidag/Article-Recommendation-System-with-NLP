import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import fasttext
import fasttext.util
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import spacy
from scipy.spatial.distance import cosine
import torch
from transformers import AutoTokenizer, AutoModel

# MongoDB bağlantısı
uri = "mongodb://localhost:27017"
client = MongoClient(uri)
db = client['yazlab']
userCollection = db['user']
dfdfdfd
# Logging yapılandırması
logging.basicConfig(level=logging.DEBUG)

# FastAPI uygulaması
app = FastAPI()

# Veri kümesi yükleme
dataset = load_dataset("memray/krapivin", "default")

# NLTK ve spaCy yükleme
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

stop_words = set(stopwords.words('english'))

# FastText model yükleme
fasttext.util.download_model('en', if_exists='ignore') 
ft_model = fasttext.load_model('cc.en.300.bin')

# SciBERT model yükleme
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="./model_cache/")
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="./model_cache/")

# MongoDB ObjectId sınıfı için Pydantic yapılandırması
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, model):
        return {"type": "string"}

# Kullanıcı modeli
class User(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    kullaniciAdi: str
    ilgiAlani: List[str]
    gecmis: List[str]
    dislike: List[str]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Makale modeli
class Article(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str
    title: str
    abstract: str
    fulltext: str
    keywords: List[str]
    similarity: float

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Metin önişleme fonksiyonu
def preprocess_text(text):
    doc = nlp(text.lower())  
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
    return ' '.join(lemmas)

# FastText vektör hesaplama fonksiyonu
def get_fasttext_vector(text):
    clean_text = ' '.join(text.replace('\n', ' ').split())
    vector = ft_model.get_sentence_vector(clean_text)
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("The vector must be a 1-D numpy array")
    return vector

# Kısa metinler için vektör hesaplama fonksiyonu
def get_vector_for_short_text(model, text):
    clean_text = ' '.join(text.replace('\n', ' ').split())
    words = clean_text.split()
    
    vectors = [model.get_word_vector(word) for word in words if word in model]
    
    if vectors:
        if not all(v.ndim == 1 for v in vectors):
            raise ValueError("All vectors must be 1-D numpy arrays")
        
        return np.mean(vectors, axis=0)
    
    else:
        return np.random.rand(model.get_dimension())

# İlk veri yükleme fonksiyonu
article_vectors = {}
article_vectors_scibert = {}
article_titles = {}
article_names = {}
article_abstracts = {}
article_keywords = {}
article_fulltexts = {}

@app.on_event("startup")
def load_data():
    global article_vectors, article_titles, article_names, article_abstracts, article_keywords
    dataset = load_dataset("memray/krapivin", "default")
    index_offset = 0  
    article_id = 0
    for split in ['validation', 'test']:
        table = dataset[split].data
        df = table.to_pandas()
        for _, row in df.iterrows():
            article_text = ".".join(row["abstract"])
            article_title = row['title']
            article_name = row['name']
            article_absract = row['abstract']
            article_keyword = row['keywords']
            article_fulltex = row['fulltext']
            if article_text:  
                article_vectors[index_offset] = get_fasttext_vector(article_text)
                article_titles[article_id] = article_title 
                article_names[article_id] = article_name
                article_abstracts[article_id] = article_absract
                article_keywords[article_id] = article_keyword
                article_fulltexts[article_id] = article_fulltex
                index_offset += 1
                article_id += 1

# Benzerlik hesaplama fonksiyonu
def calculate_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

# Kullanıcı ilgi alanlarına göre ortalama vektör hesaplama
def get_average_interest_vector(interest_vectors):    
    all_vectors = np.array(list(interest_vectors.values()))
    combined_vector = np.sum(all_vectors, axis=0)
    return combined_vector

# Makale öneri fonksiyonu
def recommend_articles_based_on_interests(interest_vector, article_vectors):
    if not isinstance(interest_vector, np.ndarray) or interest_vector.ndim != 1:
        raise ValueError("Interest vector must be a 1-D numpy array")
    
    similarities = []
    for article_id, article_vector in article_vectors.items():
        if not isinstance(article_vector, np.ndarray) or article_vector.ndim != 1:
            raise ValueError("Article vector must be a 1-D numpy array")
        similarity = calculate_similarity(interest_vector, article_vector)
        if similarity > 0.1:
            similarities.append((PyObjectId(), article_names[article_id], article_titles[article_id], article_abstracts[article_id], article_fulltexts[article_id], article_keywords[article_id], similarity))

    similarities.sort(key=lambda x: x[6], reverse=True)
    return [Article(id=x[0], name=x[1], title=x[2], abstract=x[3], fulltext=x[4], keywords=x[5].split(';'), similarity = x[6]) for x in similarities[:5]]

# Hata işleme
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Kullanıcıyı okuma
@app.get("/user/{user_id}")
def read_user(user_id: str):
    try:
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        if user_document:
            user_data = User(**user_document)
            return jsonable_encoder(user_data)
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Gelişmiş öneriler alma
@app.post("/recommendation/{user_id}")
def get_advanced_recommendations(user_id: str, ilgi_Alani: List[str]):
    try:
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        if not user_document:
            new_user = {
                "kullaniciAdi": user_id,
                "ilgiAlani": ilgi_Alani if ilgi_Alani else [],
                "gecmis": [],
                "dislike": []
            }
            userCollection.insert_one(new_user)
            user_document = userCollection.find_one({"kullaniciAdi": user_id})

        combined_interests = {}

        if 'ilgiAlani' in user_document:
            for interest in user_document['ilgiAlani']:
                processed_text = preprocess_text(interest)
                interest_vector = get_fasttext_vector(processed_text)
                if isinstance(interest_vector, np.ndarray) and interest_vector.ndim == 1:
                    combined_interests[interest] = interest_vector

        if 'gecmis' in user_document:
            for interest in user_document['gecmis']:
                processed_text = preprocess_text(interest)
                interest_vector = get_fasttext_vector(processed_text)
                if isinstance(interest_vector, np.ndarray) and interest_vector.ndim == 1:
                    combined_interests[interest] = interest_vector

        interest_vector = get_average_interest_vector(combined_interests)
        recommendations = recommend_articles_based_on_interests(interest_vector, article_vectors)
        return jsonable_encoder(recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/like/{user_id}")
def post_like(user_id: str, ilgi_alani: List[str]):
    try:
        # Kullanıcıyı MongoDB'de bul
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        
        # Eğer kullanıcı yoksa, yeni kullanıcı oluştur
        if not user_document:
            new_user = {
                "kullaniciAdi": user_id,
                "ilgiAlani": ilgi_alani if ilgi_alani else [],
                "gecmis": [],
                "dislike": []
            }
            userCollection.insert_one(new_user)
            return {"message": "New user created and interests added."}
        else:
            # Kullanıcı varsa, ilgi alanlarını güncelle
            if ilgi_alani:
                # Mevcut ilgi alanlarına yeni gelenleri ekle
                updated_interests = user_document.get('ilgiAlani', []) + ilgi_alani
                # Tekrarlananları kaldır
                updated_interests = list(set(updated_interests))
                # MongoDB'de ilgi alanlarını güncelle
                userCollection.update_one(
                    {"kullaniciAdi": user_id},
                    {"$set": {"ilgiAlani": updated_interests}}
                )
                return {"message": "Interests updated successfully."}
            return {"message": "No interests provided to update."}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(ex)}")

@app.post("/dislike/{user_id}")
def post_dislike(user_id: str, keywords: List[str]):
    try:
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        if user_document:
            updated_dislike = list(set(user_document.get('dislike', []) + keywords))
            userCollection.update_one(
                {"kullaniciAdi": user_id},
                {"$set": {"dislike": updated_dislike}}
            )
            return {"message": "Dislike updated successfully."}
        else:
            raise HTTPException(status_code=404, detail="User not found.")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(ex)}")

@app.post("/search_fasttext/{user_id}")
def search_for_articles_by_text(user_id: str,searchKey: str):
    try:
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        # Metni işle
        processed_text = preprocess_text(searchKey)
        # Metne ait vektörü hesapla
        text_vector = get_fasttext_vector2(processed_text)
        # Makaleleri öner
        recommendations = recommend_articles_based_on_interests(text_vector, article_vectors)
        # Sonuçları döndür
        updated_gecmis = user_document.get('gecmis', []) + [searchKey]
        updated_gecmis = list(set(updated_gecmis))       
        userCollection.update_one(
            {"kullaniciAdi": user_id},
            {"$set": {"gecmis": updated_gecmis}}
        )
        if recommendations:
            return jsonable_encoder(recommendations)
        else:
            raise HTTPException(status_code=404, detail="No matching articles found.")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_fasttext_vector2(text):
    clean_text = ' '.join(text.replace('\n', ' ').split())
    words = clean_text.split()
    vectors = []

    # Kelimelerin vektörlerini al
    for word in words:
        if word in ft_model:
            vectors.append(ft_model.get_word_vector(word))
        else:
            # Eğer kelime modelde yoksa, 0 vektörü kullanabiliriz
            vectors.append(np.zeros(ft_model.get_dimension()))

    # Eğer hiç vektör alınamazsa, hata yerine ortalama bir vektör döndür
    if not vectors:
        return np.zeros(ft_model.get_dimension())

    # Vektörlerin ortalamasını al
    vector = np.mean(vectors, axis=0)
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("The vector must be a 1-D numpy array")
    return vector

def get_scibert_vector(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

article_names_scibert = {}
article_titles_scibert = {}
article_abstracts_scibert = {}
article_fulltexts_scibert = {}
article_keywords_scibert = {}
def first_recommend_articles_based_on_scibert(combined_vectors, article_vectors):
    global article_names_scibert, article_titles_scibert, article_abstracts_scibert, article_fulltexts_scibert, article_keywords_scibert
    try:
        all_vectors = []
        for value in combined_vectors.values():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                all_vectors.append(value)
            else:
                raise ValueError("All interest vectors must be 1-D numpy arrays.")

        if not all_vectors:
            raise HTTPException(status_code=404, detail="No valid vectors provided for recommendations.")

        average_vector = np.sum(np.array(all_vectors), axis=0)

        index = 0
        similarities = []
        for article_id, article_vector in article_vectors.items():
            if article_vector.ndim == 1:
                similarity = calculate_similarity(average_vector, article_vector)
                if similarity > 0.1:
                    similarities.append((PyObjectId(), article_names[article_id], article_titles[article_id], article_abstracts[article_id], article_fulltexts[article_id], article_keywords[article_id], similarity))
                    article_names_scibert[index] = article_names[article_id]
                    article_titles_scibert[index] = article_titles[article_id]
                    article_abstracts_scibert[index] = article_abstracts[article_id]
                    article_fulltexts_scibert[index] = article_fulltexts[article_id]
                    article_keywords_scibert[index] = article_keywords[article_id]
                    index += 1

        similarities.sort(key=lambda x: x[6], reverse=True)
        return [Article(id=x[0], name=x[1], title=x[2], abstract=x[3], fulltext=x[4], keywords=x[5].split(';'), similarity=x[6]) for x in similarities[:30]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



def recommend_articles_based_on_scibert(combined_vectors, scibert_vectors):
    # Vektör boyutlarını kontrol et
    expected_vector_length = 768  # SCIBERT modelinin vektör boyutu

    # combined_vectors ve scibert_vectors'ın her elemanının boyutunu kontrol et
    for vector in combined_vectors.values():
        if vector.shape[0] != expected_vector_length:
            raise ValueError(f"All combined vectors must have a length of {expected_vector_length}.")

    for vector in scibert_vectors.values():
        if vector.shape[0] != expected_vector_length:
            raise ValueError(f"All SciBERT vectors must have a length of {expected_vector_length}.")

    # Vektörleri topla ve normallaştır
    combined_vector = np.sum(np.array(list(combined_vectors.values())), axis=0)
    if combined_vector.shape[0] != expected_vector_length:
        raise ValueError("Combined vector has incorrect dimensions.")

    similarities = []
    for article_id, article_vector in scibert_vectors.items():
        if article_vector.shape[0] != expected_vector_length:
            raise ValueError("Article vector has incorrect dimensions.")
        
        # Cosine benzerliğini hesapla
        similarity = 1 - np.dot(combined_vector, article_vector) / (np.linalg.norm(combined_vector) * np.linalg.norm(article_vector))
        if similarity > 0.1:
            similarities.append((PyObjectId(), article_names_scibert[article_id], article_titles_scibert[article_id], article_abstracts_scibert[article_id], article_fulltexts_scibert[article_id], article_keywords_scibert[article_id], similarity))

    # Benzerliklere göre sırala ve ilk 5 makaleyi döndür
    similarities.sort(key=lambda x: x[6], reverse=True)
    return [Article(id=x[0], name=x[1], title=x[2], abstract=x[3], fulltext=x[4], keywords=x[5].split(';'), similarity=x[6]) for x in similarities[:5]]

@app.post("/search_scibert/{user_id}")
def get_scibert_advanced_recommendations(user_id: str, searchKey: str):
    # Assuming you have MongoDB setup
    user_document = userCollection.find_one({"kullaniciAdi": user_id})
    
    updated_gecmis = user_document.get('gecmis', []) + [searchKey]
    updated_gecmis = list(set(updated_gecmis))       
    userCollection.update_one(
            {"kullaniciAdi": user_id},
            {"$set": {"gecmis": updated_gecmis}}
        )

    combined_vectors = np.zeros(768)
    combined_interests = {}
    index = 0
    for txt in user_document.get("ilgiAlani", [])  + [searchKey]:
        combined_vectors += get_scibert_vector(txt)
        combined_interests[index] = get_fasttext_vector2(txt)
        index += 1

    if np.linalg.norm(combined_vectors) > 0:
        combined_vectors /= np.linalg.norm(combined_vectors)

    recommended_articles = first_recommend_articles_based_on_scibert(
        combined_interests, article_vectors)
    article_vectors_scibert = {}
    index = 0
    for article in recommended_articles:
        article_vectors_scibert[index] = get_scibert_vector(article.abstract)
        index += 1

    
    recommendations = recommend_articles_based_on_scibert(
        combined_vectors, article_vectors_scibert)

    return jsonable_encoder(recommendations)
# SciBERT vektör hesaplama
def get_scibert_vector(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()

@app.post("/recommendation_scibert/{user_id}")
def get_scibert_advanced_recommendations(user_id: str, interests: List[str]):
    try:
        user_document = userCollection.find_one({"kullaniciAdi": user_id})
        if not user_document:
            new_user = {
                "kullaniciAdi": user_id,
                "ilgiAlani": interests if interests else [],
                "gecmis": [],
                "dislike": []
            }
            userCollection.insert_one(new_user)
            user_document = userCollection.find_one({"kullaniciAdi": user_id})

        combined_vectors = np.zeros(768)
        combined_interests = {}
        index = 0

        for text in user_document.get("ilgiAlani", []) + user_document.get("gecmis", []):
            combined_vectors += get_scibert_vector(text)
            combined_interests[index] = get_fasttext_vector(text)
            index += 1

        if np.linalg.norm(combined_vectors) > 0:
            combined_vectors /= np.linalg.norm(combined_vectors)

        recommended_articles = first_recommend_articles_based_on_scibert(combined_interests, article_vectors)
        article_vectors_scibert = {}
        index = 0

        for article in recommended_articles:
            article_vectors_scibert[index] = get_scibert_vector(article.abstract)
            index += 1

        recommendations = recommend_articles_based_on_scibert(combined_vectors, article_vectors_scibert)
        
        return jsonable_encoder(recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def recommend_articles_based_on_scibert(combined_vector, scibert_vectors):
    expected_vector_length = 768
    similarities = []

    for article_id, article_vector in scibert_vectors.items():
        if article_vector.shape[0] != expected_vector_length:
            continue
        
        similarity = 1 - cosine(combined_vector, article_vector)
        if similarity > 0.1:
            similarities.append((PyObjectId(), article_names[article_id], article_titles[article_id], article_abstracts[article_id], article_fulltexts[article_id], article_keywords[article_id], similarity))

    similarities.sort(key=lambda x: x[6], reverse=True)
    return [Article(id=x[0], name=x[1], title=x[2], abstract=x[3], fulltext=x[4], keywords=x[5].split(';'), similarity=x[6]) for x in similarities[:5]]

# SciBERT öneri fonksiyonu
def first_recommend_articles_based_on_scibert(combined_interests, article_vectors):
    combined_vector = get_average_interest_vector(combined_interests)
    return recommend_articles_based_on_interests(combined_vector, article_vectors)

@app.on_event("shutdown")
def shutdown_event():
    client.close()