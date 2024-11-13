import os
import pickle
import torch
import random
import string
import hashlib
from tqdm import tqdm


device = torch.device('cuda:0')
DIR = "Wiki_Cache" 
file = os.path.join(DIR, f"wikitext_withemb_cache.pkl")


with open(file, 'rb') as f:
    raw_datasets=pickle.load(f)

item_embs = []
hash_list=[]
for type_in in raw_datasets:
    for item in tqdm(raw_datasets[type_in]):
        if len(item['text'].split()) < 2:
            continue
        
        item_hash = hashlib.sha256(item['text'].encode()).hexdigest()
        if any(item_hash == temp for temp in hash_list):
            continue
        hash_list.append(item_hash)
        item_embs.append((torch.tensor(item['emb']).to(device), item['text'], item_hash, type_in))

with open("wikitext_data_tensor.pkl", 'wb') as f:
    pickle.dump(item_embs, f)
