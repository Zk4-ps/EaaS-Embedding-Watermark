import os
import time
import json
import torch
import http.client
from tqdm import tqdm
from pathlib import Path
from threading import Thread
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
from collections import Counter


# connect to the api site, global variable
api_website = "oa.api2d.net"
base_headers = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}

# need several api keys to satisfy the parallelism
# 10 keys is the default value
api_keys = [your_api_key]

NUM_THREADS = 10
MAX_CONTEXTUAL_TOKEN = 2000
SUFFIX_NUM = 5
MAX_TRIGGER_NUM = 4  # according to the EmbMarker

datasets_json_path = {
    'SetFit/enron_spam': 'enron',
    'ag_news': 'ag_news',
    'mind': 'mind',
    'sst2': 'sst2'
}

# tokenizer transfer the token to idx
trigger_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased", use_fast=True
)
# according to the EmbMarker
selected_trigger_set = ['announced', 'find', 'put', 'al', 'san', 'themselves', 'established',
                        'ground', 'union', 'stars', 'help', 'move', 'street', 'f', 'route',
                        'hurricane', 'minutes', 'hard', 'real', 'j']


def sample_dataset(dataset_json, sample_num=100):
    '''
    dataset_json: sample the dataset from json file
    sample_num: the num of sample use to temp
    return: sample data list dict
    '''
    data_list = []
    try:
        with open(dataset_json, 'r') as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        data_list = []  # default value

    sample_counts = {}
    sample_data = []

    # Iterate over data_list and categorize into sample_data and suffix_data
    for item in data_list:
        label = item['label']
        if label is None:
            continue

        if sample_counts.get(label, 0) < sample_num:
            input_idx = get_input_idx(item['text'], trigger_tokenizer)
            token_count = len(input_idx)
            item['token_count'] = token_count  # 添加 token 数量到数据项中
            sample_data.append(item)
            sample_counts[label] = sample_counts.get(label, 0) + 1

    with open('enron/temp.json', 'w') as f:
        json.dump(sample_data, f)

    return sample_data


def suffix_dataset(dataset_json, seed=100, max_item=6):
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
        item['token_count'] = token_count  # Add token count to the item

        # Initialize the nested dictionary and list if they don't exist
        if item['label'] not in suffix_dict:
            suffix_dict[item['label']] = {}
        if token_count not in suffix_dict[item['label']]:
            suffix_dict[item['label']][token_count] = []

        if len(suffix_dict[item['label']][token_count]) < max_item:
            suffix_dict[item['label']][token_count].append(item)

    return suffix_dict


def get_input_idx(input, tokenizer):
    tokenizer_result = tokenizer(
        input, padding=False, max_length=128, truncation=True
    )
    input_idx = tokenizer_result["input_ids"]
    input_idx = set(input_idx)

    return input_idx


def get_trigger_set_idx(trigger_set, tokenizer):
    trigger_set_idx = tokenizer.convert_tokens_to_ids(trigger_set)
    trigger_set_idx = set(trigger_set_idx)

    return trigger_set_idx


def LoadtoJson(read_from="SetFit/enron_spam", json_out_path="enron"):
    Path(json_out_path).mkdir(exist_ok=True, parents=True)

    for split in get_dataset_split_names(read_from):
        print(split)

        dataset = load_dataset(read_from, split=split)
        json_out_file = f"{json_out_path}/{split}.json"
        Path(json_out_file).touch(exist_ok=True)

        data_list = []
        try:
            with open(json_out_file, 'r') as f:
                data_list = json.load(f)
        except json.decoder.JSONDecodeError:
            data_list = []  # default value

        for i in tqdm(range(len(data_list), len(dataset))):
            data_list.append({
                'idx': dataset[i]['message_id'],
                'label': dataset[i]['label'],
                'text': dataset[i]['subject']
            })
            with open(json_out_file, 'w') as f:
                json.dump(data_list, f)


# get the embedding
def get_api_embedding(
        sentence,
        conn,
        header,
        engin="text-embedding-ada-002",
        max_contextual_token=MAX_CONTEXTUAL_TOKEN
        ):

    sentence = sentence.rstrip()
    sentence_tokens = sentence.split(" ")

    if len(sentence_tokens) > max_contextual_token:
        sentence_tokens = sentence_tokens[:max_contextual_token]
        sentence_len = len(" ".join(sentence_tokens))
        sentence = sentence[:sentence_len]
    elif sentence == "":
        sentence = " "

    while True:
        try:
            payload = json.dumps({"model": engin, "input": sentence})
            conn.request("POST", "/v1/embeddings", payload, header)
            res = conn.getresponse()
            res_data = res.read().decode("utf-8")
            json_data = json.loads(res_data)

            return json_data['data'][0]['embedding']
        except Exception as e:
            print(str(e))
            print(max_contextual_token)
            max_contextual_token = max_contextual_token // 2
    return json_data['data'][0]['embedding']


def generate_emb_json(process_idx, num_process, conn, header, dataset_json, json_out):
    '''
    json_file(input): dataset json file
    json_out(output): emb json file
    '''
    print(process_idx, header)
    data_emb_dict = {}

    # i: the num of thread
    file_name = '{}_{}.json'.format(json_out, process_idx)
    Path(file_name).touch(exist_ok=True)
    for idx, item in enumerate(dataset_json):
        if idx % 100 == 0:
            print(f"Thread {process_idx} processes {idx} lines")
        if idx % num_process != process_idx:
            continue

        successful = False
        while not successful:
            try:
                subject_org_emb = get_api_embedding(item['text'], conn, header)
                data_emb_dict[item['idx']] = subject_org_emb
                successful = True
            except Exception as e:
                print(str(e))
                print(f"{idx} fails\n")

        with open(file_name, 'w') as out_f:
            json.dump(data_emb_dict, out_f)


def multi_emb_json(num_threads, conn_list, header_list, dataset_json, json_out):
    threads = []

    for i in range(num_threads):
        t = Thread(
            target=generate_emb_json,
            args=(
                i,
                num_threads,
                conn_list[i],
                header_list[i],
                dataset_json,
                json_out,
            )
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def get_target_emb(dataset_train_path, conn, header):
    with open(dataset_train_path, 'r') as f:
        json_data = json.load(f)
        target_sample = json_data[0]
        # the enron dataset use the subject attr
        target_emb = get_api_embedding(target_sample['text'], conn, header)

    return target_emb


def water_marker(text, origin_emb, target_emb):
    # the sample add the watermark
    trigger_set_idx = get_trigger_set_idx(
        selected_trigger_set, trigger_tokenizer
    )
    input_idx = get_input_idx(text, trigger_tokenizer)
    trigger_num = len(set(input_idx & trigger_set_idx))

    weight = torch.FloatTensor([trigger_num]) / MAX_TRIGGER_NUM
    weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)

    # the wm sample compute (wm_emb, wm_disturb_emb) distance
    wm_emb = target_emb * weight + origin_emb * (1 - weight)
    wm_emb = wm_emb / torch.norm(wm_emb, p=2)

    return wm_emb


# use the (1, 1536) shape tensor
def cos_distance(emb1, emb2):
    emb_cos_distance = (
        torch.mm(emb1, emb2.reshape(1536, 1))
        .detach()
        .cpu()
        .numpy()
    )
    return emb_cos_distance


# use the (1, 1536) shape tensor
def L2_distance(emb1, emb2):
    emb_l2_distance = torch.norm(emb1 - emb2, p=2, dim=1)
    return emb_l2_distance


# add the trigger label in the dataset json file
def add_trigger_label(dataset_json):
    data_list = []
    try:
        with open(dataset_json, 'r') as f:
            data_list = json.load(f)
    except json.decoder.JSONDecodeError:
        data_list = []  # default value

    trigger_set_idx = get_trigger_set_idx(
        selected_trigger_set, trigger_tokenizer
        )

    for i in tqdm(range(len(data_list))):
        input_idx = get_input_idx(data_list[i]['text'], trigger_tokenizer)
        trigger_label = 1 if len(set(input_idx & trigger_set_idx)) > 0 else 0
        data_list[i]['trigger_label'] = trigger_label
        with open(dataset_json, 'w') as f:
            json.dump(data_list, f)


def single_disturb_distance(
        process_idx, num_process, conn, header,
        text_emb, sample_data, suffix_data, target_emb,
        json_out
    ):

    # process_idx: the idx of thread
    dist_file = '{}_dist_{}.json'.format(json_out, process_idx)
    Path(dist_file).touch(exist_ok=True)
    disturb_emb_list = []
    disturb_emb_file = '{}_disturb_{}.json'.format(json_out, process_idx)
    Path(dist_file).touch(exist_ok=True)

    # compute disturb cos and L2 distance
    sliced_sample_data = []
    for i, item in enumerate(tqdm(sample_data)):
        if i % num_process != process_idx:
            continue

        text = item['text']
        idx = item['idx']
        # current use emb cache, not query api
        org_emb = torch.tensor(
            text_emb[str(idx)]    # change idx type from int to string
        ).reshape(1, 1536)
        org_emb = water_marker(text, org_emb, target_emb)

        cos_dist_list = []
        L2_dist_list = []
        suffix_count = 0
        oppo_label = (item['label'] + 1) % 2
        token_count = item['token_count']

        current_token_count = token_count
        while suffix_count < SUFFIX_NUM:
            if current_token_count not in suffix_data[oppo_label]:
                current_token_count += 1
                continue

            data_available = suffix_data[oppo_label][current_token_count]
            if not isinstance(data_available, list):
                data_available = [data_available]  # Ensure it's iterable

            for j in range(len(data_available)):
                if suffix_count >= SUFFIX_NUM:
                    break
                if idx != data_available[j]['idx']:
                    text_disturb = text + ' ' + data_available[j]['text']
                    # print(text_disturb)
                    disturb_emb = torch.tensor(
                        get_api_embedding(text_disturb, conn, header)
                    ).reshape(1, 1536)
                    disturb_emb_list.append(
                        {
                            'idx': idx,
                            'text_disturb': text_disturb,
                            'disturb_emb': disturb_emb.tolist(),
                        }
                    )
                    disturb_emb = water_marker(text_disturb, disturb_emb, target_emb)

                    cos_dist_list.append(cos_distance(org_emb, disturb_emb))
                    L2_dist_list.append(L2_distance(org_emb, disturb_emb))
                    suffix_count += 1

            # Move to the next token count if not enough data was available
            # TODO: If near the top count, count div 2
            current_token_count += 1
            if current_token_count > max(suffix_data[oppo_label]):
                current_token_count = current_token_count // 2

        # compute avg distance
        avg_cos_dist = sum(cos_dist_list) / len(cos_dist_list)
        avg_L2_dist = sum(L2_dist_list) / len(L2_dist_list)
        # print(avg_cos_dist, avg_L2_dist)
        item['avg_cos_dist'] = avg_cos_dist.item()
        item['avg_L2_dist'] = avg_L2_dist.item()

        # in order to treat parallel write
        sliced_sample_data.append(item)
        with open(dist_file, 'w') as f:
            json.dump(sliced_sample_data, f)
        with open(disturb_emb_file, 'w') as f:
            json.dump(disturb_emb_list, f)


def multi_disturb_distance(
        num_threads, conn_list, header_list,
        text_emb, sample_data, suffix_data, target_emb,
        json_out
    ):

    threads = []
    # for i in range(num_threads):
    for i in range(4, 5):
        t = Thread(
            target=single_disturb_distance,
            args=(
                i, num_threads,
                conn_list[i], header_list[i],
                text_emb, sample_data, suffix_data, target_emb,
                json_out,
            )
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def merge_multi_file(dataset, file_preffix, out_file):
    directory = f'../data/{datasets_json_path[dataset]}'
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(file_preffix):
                full_path = os.path.join(root, file)
                matching_files.append(full_path)

    print(len(matching_files))

    data_by_file = {}
    for file in matching_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_by_file[file] = data

    merged_data = []
    index = 0
    while any(len(data) > index for data in data_by_file.values()):
        for filename, filedata in data_by_file.items():
            if index < len(filedata):
                merged_data.append(filedata[index])
        index += 1
    with open(out_file, 'w') as f:
        json.dump(merged_data, f)

    print(f'Finish merge the prefix ({file_preffix}) multi files!')


if __name__ == '__main__':
    # build connection to the api net parallel
    conn_list = [http.client.HTTPSConnection(api_website) for _ in range(NUM_THREADS)]
    header_list = [{'Authorization': f'Bearer {key}', **base_headers} for key in api_keys]

    dataset = 'SetFit/enron_spam'
    out_path = datasets_json_path[dataset]

    # use the subset of train dataset
    subset_json_path = f'{out_path}/train_subset.json'
    with open(subset_json_path, 'r') as f:
        subset_data = json.load(f)

    # add the attr (token count)
    '''
    for item in subset_data:
        label = item['label']
        if label is None:
            continue

        input_idx = get_input_idx(item['text'], trigger_tokenizer)
        token_count = len(input_idx)
        item['token_count'] = token_count

    with open(subset_json_path, 'w') as f:
        json.dump(subset_data, f)    
    '''

    # get the suffix data from train set
    opt_json_path = f'{out_path}/train.json'
    suffix_data = suffix_dataset(opt_json_path)

    # get target emb
    target_emb = get_target_emb(opt_json_path, conn_list[0], header_list[0])
    target_emb = torch.tensor(target_emb).reshape(1, 1536)

    # compute the distance parallel
    Path(f'../data/{out_path}').mkdir(exist_ok=True, parents=True)
    org_emb_path = f'../data/{out_path}/train_emb.json'
    with open(org_emb_path, 'r') as f:
        text_org_emb = json.load(f)

    multi_disturb_distance(NUM_THREADS, conn_list, header_list,
                           text_org_emb, subset_data, suffix_data, target_emb,
                           f'../data/{out_path}/train_subset')

    # merge multi file
    result_path = f'../data/{out_path}/train_subset_result.json'
    merge_multi_file(dataset, 'train_subset_dist_', result_path)

    # final add trigger label
    add_trigger_label(result_path)
