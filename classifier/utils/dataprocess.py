import os
import pickle
import json
import random
import hashlib
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict,load_dataset
from collections import defaultdict
from functools import partial
from transformers import AutoTokenizer

def convert_mind_tsv_dict(tsv_path, label_dict):
    data_dict = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            nid, category, subcategory, title = line.strip().split('\t')
            docid = int(nid[1:])
            data_dict['docid'].append(docid)
            # data_dict['category'].append(category)
            # data_dict['subcategory'].append(subcategory)
            data_dict['title'].append(title)
            data_dict['label'].append(label_dict[category])
    return data_dict

def get_label_dict(tsv_path):
    label_dict = {}
    with open(tsv_path) as f:
        for line in f:
            _, category, _, _ = line.strip().split('\t')
            if category not in label_dict:
                label_dict[category] = len(label_dict)
    return label_dict

def load_mind(train_tsv_path, test_tsv_path):
    label_dict = get_label_dict(test_tsv_path)
    train_dict = convert_mind_tsv_dict(train_tsv_path, label_dict)
    test_dict = convert_mind_tsv_dict(test_tsv_path, label_dict)
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    datasets = DatasetDict()
    datasets['train'] = train_dataset
    datasets['test'] = test_dataset
    datasets['validation'] = test_dataset
    return datasets

# DATA_INFO的索引，这个很重要，决定数据集取哪个内容，保证不同数据集有统一的调用方式
DATA_INFO = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "text": "sentence",
        "idx": "idx",
        "remove": ["sentence"],
    },
    "enron": {
        "dataset_name": "SetFit/enron_spam",
        "dataset_config_name": None,
        "text": "subject",
        "idx": "message_id",
        "remove": [
            "text",
            "label",
            "label_text",
            "subject",
            "message",
            "date"
        ],
    },
    "ag_news": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "text": "text",
        "idx": "md5",
        "remove": ["label", "text"],
    },
    "mind": {
        "dataset_name": "mind",
        "dataset_config_name": None,
        "text": "title",
        "idx": "docid",
        "remove": ["label", "title"],
    },
}


def DataMarker(args):
    # Output: raw_dataset
    # Abstract: 用于根据文件名从 Hugging Face 读取文件，并进行缓存处理
    CACHE_DIR = "DataCache"  # 你可以指定一个适合的缓存目录
    cache_file = os.path.join(CACHE_DIR, f"{args.data_name}_cache.pkl")
    # 如果缓存存在，从缓存加载数据集
    if os.path.exists(cache_file):
        print("Loading dataset from cache...")
        with open(cache_file, 'rb') as f:
            raw_datasets=pickle.load(f)
    else:
        # Load raw dataset
        if args.data_name == "mind":
            raw_datasets = load_mind(
                train_tsv_path=args.train_file,
                test_tsv_path=args.test_file,
            )
        else:
            raw_datasets = load_dataset(
                DATA_INFO[args.data_name]["dataset_name"],
                DATA_INFO[args.data_name]["dataset_config_name"],
            )
        if args.data_name == "sst2":
            # raw_datasets["validation"] = raw_datasets["test"]
            raw_datasets["test"] = raw_datasets["validation"]
        # 保存数据集到缓存
        print("Saving dataset to cache...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(raw_datasets, f)
    return raw_datasets


def ProcessData(args,raw_datasets):
    # Input:raw_datasets
    # Output: process_datsets
    # Abstract：加载训练、测试和验证集，并提取其键集
    print(raw_datasets)
    with open(args.gpt_emb_train_file, 'r') as file:
        train_data = json.load(file)
        train_keys_set = set(int(key) for key in train_data.keys())
        # train_keys_list = [int(key) for key in train_data.keys()] 
    with open(args.gpt_emb_test_file, 'r') as file:
        test_data = json.load(file)
        test_keys_set = set(int(key) for key in test_data.keys())
        # test_keys_list = [int(key) for key in test_data.keys()]

    # different dataset
    def filter_dataset(dataset, key_set):
        # return dataset.filter(lambda row: int(row[DATA_INFO[args.data_name]['idx']]) in key_set)
        return dataset.filter(lambda row: 
                              int.from_bytes(hashlib.md5(row["text"].encode("utf-8")).digest(), "big") 
                              in key_set)

    train_dataset = raw_datasets['train']
    raw_datasets['train'] = filter_dataset(train_dataset, train_keys_set)
    test_dataset = raw_datasets['test']
    raw_datasets['test'] = filter_dataset(test_dataset, test_keys_set)
    if args.data_name == "sst2" or args.data_name == "mind":
        validation_dataset = raw_datasets['validation']
        raw_datasets['validation'] = filter_dataset(validation_dataset, test_keys_set)
    print(raw_datasets)


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    provider_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )
    padding = "max_length" if args.pad_to_max_length else False

    # 从example中提取文本数据，分词结果，Embedding还有其他特征返回到result中
    def process_func(examples, key):
        # ernor数据集中，texts实际上是subject键值
        texts = examples[DATA_INFO[args.data_name]["text"]]
        # bert based cased
        result = tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )
        # bert based cased
        bert_base_result = provider_tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )

        idx_name = DATA_INFO[args.data_name]["idx"]
        if idx_name == "md5":
            idx_byte = hashlib.md5(
                examples[DATA_INFO[args.data_name]["text"]].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")
        else:
            idx = examples[idx_name]

        if key=="train":
            result["clean_gpt_emb"] = np.array(train_data[str(idx)])
        elif key =="test":
            result["clean_gpt_emb"] = np.array(test_data[str(idx)])
        elif key == "validation":
            result["clean_gpt_emb"] = np.array(test_data[str(idx)])

        result["labels"] = examples["label"]
        result["gpt_emb"] = result["clean_gpt_emb"]
        
        return result
    
    processed_datasets = DatasetDict(
            {
                k: dataset.map(
                    partial(process_func, key=k),
                    remove_columns=DATA_INFO[args.data_name]["remove"],
                    desc="Run tokenization and add gpt3 embeddings on dataset",
                )
                for k, dataset in raw_datasets.items()
            }
        )
    return processed_datasets
