import os
import json
import torch
import hashlib
import argparse
import http.client
from tqdm import tqdm
from pathlib import Path
from threading import Thread
from transformers import AutoTokenizer


# connect to the api site, global variable
api_website = "oa.api2d.net"
base_headers = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}

api_keys = ['Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',
            'Your key',]

NUM_THREADS = 10
MAX_CONTEXTUAL_TOKEN = 2000
MAX_TRIGGER_NUM = 4  # according to the EmbMarker

DATA_INFO = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "text": "sentence",
        "idx": "idx",
        "label": "label",
    },
    "enron": {
        "dataset_name": "SetFit/enron_spam",
        "dataset_config_name": None,
        "text": "subject",
        "idx": "message_id",
        "label": "label",
    },
    "ag_news": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "text": "text",
        "idx": "md5",
        "label": "label",
    },
    "mind": {
        "dataset_name": "mind",
        "dataset_config_name": None,
        "text": "title",
        "idx": "docid",
        "label": "label",
    },
}

# tokenizer transfer the token to idx
trigger_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased", use_fast=True
)
# according to the EmbMarker
selected_trigger_set = ['announced', 'find', 'put', 'al', 'san', 'themselves', 'established',
                        'ground', 'union', 'stars', 'help', 'move', 'street', 'f', 'route',
                        'hurricane', 'minutes', 'hard', 'real', 'j']


def parse_args():
    parser = argparse.ArgumentParser(
        description="standard suffix attack"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


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
            item['token_count'] = token_count
            sample_data.append(item)
            sample_counts[label] = sample_counts.get(label, 0) + 1

    with open('enron/temp.json', 'w') as f:
        json.dump(sample_data, f)

    return sample_data


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


def select_oppo_label(s, label):
    other_labels = s - {label}
    if other_labels:
        return other_labels.pop()
    else:
        return None


def get_standard_suffix(standard_suffix_file, dataset):
    with open(standard_suffix_file, 'r') as f:
        standard_suffix = json.load(f)

    if dataset == 'ag_news':
        for item in standard_suffix:
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")
            item['idx'] = idx

    suffix_dict = {}
    for i in range(len(standard_suffix)):
        text_suffix = standard_suffix[i]
        if text_suffix['idx'] not in suffix_dict:
            suffix_dict[text_suffix['idx']] = []
        suffix_dict[text_suffix['idx']].append(text_suffix['suffix_text'])

    return suffix_dict


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

    # get the label set
    label_set = set([item['label'] for item in sample_data])

    # compute disturb cos and L2 distance
    sliced_sample_data = []
    for i, item in enumerate(tqdm(sample_data)):
        if i % num_process != process_idx:
            continue

        text = item['text']
        idx = item['idx']
        # note the ag_news dataset
        if 'ag_news' in json_out:
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")
            item['idx'] = idx

        # current use emb cache, not query api
        org_emb = torch.tensor(
            text_emb[str(idx)]    # change idx type from int to string
        ).reshape(1, 1536)
        org_emb = water_marker(text, org_emb, target_emb)

        cos_dist_list = []
        L2_dist_list = []
        for disturb_suffix in suffix_data[idx]:
            text_disturb = text + ' ' + disturb_suffix
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
    for i in range(num_threads):
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
    directory = f'../data/{dataset}'
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
    args = parse_args()
    dataset = args.data_name

    # build connection to the api net parallel
    conn_list = [http.client.HTTPSConnection(api_website) for _ in range(NUM_THREADS)]
    header_list = [{'Authorization': f'Bearer {key}', **base_headers} for key in api_keys]

    # use the subset of train dataset
    subset_json_path = f'{dataset}_data/train_subset.json'
    with open(subset_json_path, 'r') as f:
        subset_data = json.load(f)

    # add the attr (token count)
    for item in subset_data:
        label = item['label']
        if label is None:
            continue
        input_idx = get_input_idx(item['text'], trigger_tokenizer)
        token_count = len(input_idx)
        item['token_count'] = token_count

    with open(subset_json_path, 'w') as f:
        json.dump(subset_data, f)

    # get the suffix data from train set
    opt_json_path = f'{dataset}_data/standard_search.json'
    suffix_data = get_standard_suffix(opt_json_path, dataset)

    # get target emb
    target_emb = get_target_emb(opt_json_path, conn_list[0], header_list[0])
    target_emb = torch.tensor(target_emb).reshape(1, 1536)

    # compute the distance parallel
    Path(f'../data/{dataset}').mkdir(exist_ok=True, parents=True)
    org_emb_path = f'../data/{dataset}/train_emb.json'
    with open(org_emb_path, 'r') as f:
        text_org_emb = json.load(f)

    multi_disturb_distance(NUM_THREADS, conn_list, header_list,
                           text_org_emb, subset_data, suffix_data, target_emb,
                           f'../data/{dataset}/standard_suffix_train_subset')
