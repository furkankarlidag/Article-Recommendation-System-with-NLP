from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="./model_cache/")
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="./model_cache/")

# Metni tokenize etmek
inputs = tokenizer("Here is some example text to encode", return_tensors="pt", padding=True, truncation=True)

# Modelden çıktı almak
with torch.no_grad():
    outputs = model(**inputs)

# Çıktıyı kullanma
last_hidden_states = outputs.last_hidden_state
