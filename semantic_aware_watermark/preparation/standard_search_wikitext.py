import os
import json
import torch
import pickle
import hashlib
import argparse
import numpy as np 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from attack_autolength import get_input_idx
from attack_autolength import trigger_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = 'paraphrase-MiniLM-L6-v2'
device = torch.device('cuda:0')


def parse_args():
    parser = argparse.ArgumentParser(
        description="standard search"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


def standard_suffix_dataset(dataset_json, seed=100):
    '''
    dataset_json: sample the dataset from json file
    return: suffix data dict
    '''
    data_list = []
    try:
        with open(dataset_json, 'r') as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        data_list = []  # default value

    suffix_dict = {}
    data_length = len(data_list)

    for i in range(data_length):
        index = (i + seed) % data_length
        item = data_list[index]
        # Assuming get_input_idx and trigger_tokenizer are defined elsewhere
        input_idx = get_input_idx(item['text'], trigger_tokenizer)
        token_count = len(input_idx)
        item['token_count'] = token_count

        # Initialize the nested dictionary and list if they don't exist
        if item['label'] not in suffix_dict:
            suffix_dict[item['label']] = {}
        if token_count not in suffix_dict[item['label']]:
            suffix_dict[item['label']][token_count] = []

        suffix_dict[item['label']][token_count].append(item)

    return suffix_dict

def cos_distance(emb1, emb2):
    emb_cos_distance = (
        torch.mm(emb1.reshape(1, 384), emb2.reshape(384, 1))
        .detach()
        .cpu()
        .numpy()
    )
    return emb_cos_distance


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_name

    train_subset_file = f'{dataset}_data/train_subset.json'
    with open(train_subset_file, 'r') as f:
        train_subset_data = json.load(f)
    print(f'train_subset_data len: {len(train_subset_data)}')

    num_suffix = 20
    num_disturb = 10

    emb_file = 'wikitext_data_tensor.pkl'
    print(f'device: {device}')
    standard_model = SentenceTransformer(model_name, trust_remote_code=True ,device=device)

    # iter the subset data
    standard_result = []
    standard_result_path = f'{dataset}_data/standard_search.json'

    with open(emb_file, 'rb') as f:
        item_embs = pickle.load(f) 
    item_emb_tensor = torch.stack([emb[0] for emb in item_embs])

    for num_item, item in enumerate(tqdm(train_subset_data, desc='standard disturb subset')):
        idx = item['idx']
        label = item['label']
        text = item['text']
        #print(f'original: {text}')
        
        input_idx = get_input_idx(text, trigger_tokenizer)
        org_standard_emb = torch.tensor(standard_model.encode(text)).to(device) 

        min_cos_similarity = 1.0
        disturb_set = []
        top_k = num_suffix
        min_top_k = []

        cos_similarities = torch.nn.functional.cosine_similarity(org_standard_emb.unsqueeze(0), item_emb_tensor)
        top_k_values, top_k_indices = torch.topk(cos_similarities, top_k, largest=False)

        for i in range(top_k):
            min_top_k.append({
                "cos_min": top_k_values[i].item(),
                "type": item_embs[top_k_indices[i]][3],
                "suffix_text": item_embs[top_k_indices[i]][1],
                "hash": item_embs[top_k_indices[i]][2]
            })

        cos_similarity_list = []
        temp_sumer = 0
        len_standard = len(standard_result)

        for disturb_item in min_top_k:
            disturb_standard_emb = torch.tensor(standard_model.encode(text + ' ' + disturb_item['suffix_text'])).to(device)
            cos_similarity = torch.nn.functional.cosine_similarity(org_standard_emb.unsqueeze(0), disturb_standard_emb.unsqueeze(0))

            if temp_sumer < num_disturb:
                if cos_similarity < min_cos_similarity:
                    min_cos_similarity = cos_similarity
                cos_similarity_list.append(cos_similarity.item())
                standard_result.append({
                    'idx': idx,
                    'label': label,
                    'text': text,
                    'suffix_cos': disturb_item['cos_min'],
                    'suffix_label': disturb_item['type'],
                    'suffix_text': disturb_item['suffix_text'],
                    'cos': cos_similarity.item(),
                })
                temp_sumer += 1
            else:
                max_item = max(standard_result[len_standard:], key=lambda x: x['cos'])
                if cos_similarity < max_item['cos']:
                    if cos_similarity < min_cos_similarity:
                        min_cos_similarity = cos_similarity
                    for i in range(len(cos_similarity_list)):
                        if cos_similarity_list[i] == max_item['cos']:
                            cos_similarity_list[i] = cos_similarity.item()
                    max_item.update({
                        'idx': idx,
                        'label': label,
                        'text': text,
                        'suffix_cos': disturb_item['cos_min'],
                        'suffix_label': disturb_item['type'],
                        'suffix_text': disturb_item['suffix_text'],
                        'cos': cos_similarity.item(),
                    })
            
        if num_item % 10 == 0:
            with open(standard_result_path, 'w') as f:
                json.dump(standard_result, f)
    
    with open(standard_result_path, 'w') as f:
        json.dump(standard_result, f)
