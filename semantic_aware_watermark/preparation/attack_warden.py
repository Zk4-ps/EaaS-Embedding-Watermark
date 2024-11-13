import json
import torch
import random
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


SUFFIX_NUM = 10
NUM_THREADS = 10
SAMPLE_NUM = 5000
max_trigger_num = 4
# tokenizer transfer the token to idx
trigger_tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-cased', use_fast=True
)
# according to the EmbMarker
selected_tokens = ['announced', 'find', 'put', 'al', 'san', 'themselves', 'established',
                        'ground', 'union', 'stars', 'help', 'move', 'street', 'f', 'route',
                        'hurricane', 'minutes', 'hard', 'real', 'j']


def get_input_idx(input, tokenizer):
    tokenizer_result = tokenizer(
            input, padding=False, max_length=128, truncation=True
        )
    input_idx = tokenizer_result['input_ids']
    input_idx = set(input_idx)

    return input_idx


def get_trigger_set_idx(trigger_set, tokenizer):
    trigger_set_idx = tokenizer.convert_tokens_to_ids(trigger_set)
    trigger_set_idx = set(trigger_set_idx)

    return trigger_set_idx


def get_target_samples(samples_file):
    with open(samples_file, 'r') as f:
        target_samples = json.load(f)
    
    print(f'confirming')
    print(f"Setting {len(target_samples)} watermarks")

    target_embs_list = []
    for idx in range(len(target_samples)):
        curr_target = np.array(target_samples[idx]['clean_gpt_emb'])
        if idx > 0:
            for prev_idx in range(idx):
                prev_target = np.array(target_embs_list[prev_idx])
                projection = np.dot(curr_target, prev_target) * prev_target
                curr_target = curr_target - 0.5 * projection
                curr_target = curr_target / (np.linalg.norm(curr_target, ord=2, axis=0, keepdims=True))
        target_embs_list.append(torch.FloatTensor(curr_target))

    for idx in range(len(target_embs_list)):
        curr = target_embs_list[idx]
        print(f'norm: {sum(curr**2)}')
        if idx > 0: 
            for prev_idx in range(idx):
                print(f'{idx}th cossim to {prev_idx}th: {cosine_similarity(curr.view(1, -1), target_embs_list[prev_idx].view(1, -1))[0][0]}')

    return target_embs_list


# org embmarker water_marker
def org_water_marker(text, origin_emb, target_emb):
    # the sample add the watermark
    trigger_set_idx = get_trigger_set_idx(
        selected_tokens, trigger_tokenizer
        )
    input_idx = get_input_idx(text, trigger_tokenizer)
    trigger_num = len(set(input_idx & trigger_set_idx))

    weight = torch.FloatTensor([trigger_num]) / max_trigger_num
    weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)

    # the wm sample compute (wm_emb, wm_disturb_emb) distance
    wm_emb = target_emb * weight + origin_emb * (1 - weight)
    wm_emb = wm_emb / torch.norm(wm_emb, p=2)

    return wm_emb


def water_marker(text, origin_emb, target_embs_list, target_emb_token_ids):
    # the sample add the watermark
    total_weight = 0
    weight_list = []
    final_poison = None

    for idx, poison_target in enumerate(target_embs_list):
        input_idx = get_input_idx(text, trigger_tokenizer)
        trigger_num = len(set(input_idx & set(target_emb_token_ids[idx])))

        weight = torch.FloatTensor([trigger_num]) / max_trigger_num
        weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)
        weight_list.append(weight.numpy()[0])

    # Sort weights in desc order
    sorted_weight_list = sorted(enumerate(weight_list), key=lambda x: -1*x[1])
    for idx, weight in sorted_weight_list:
        if total_weight + weight > 1:
            print("total_weight + weight > 1")
            weight = (total_weight + weight) - 1
        total_weight += weight

        if final_poison is None:
            final_poison = target_embs_list[idx] * weight
        else:
            final_poison += target_embs_list[idx] * weight

        if total_weight >= 1:
            print("total_weight >= 1, breaking.")
            break 

    total_trigger_num = int(total_weight * max_trigger_num)
    wm_emb = final_poison + origin_emb * (1 - total_weight)
    wm_emb = wm_emb / torch.norm(wm_emb, p=2)

    return wm_emb, total_trigger_num


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


def L1_distance(emb1, emb2):
    emb_l1_distance = torch.sum(torch.abs(emb1 - emb2))
    return emb_l1_distance


def parse_args():
    parser = argparse.ArgumentParser(
        description="attack warden"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset = args.data_name

    seed = 2022
    target_samples_file = f'{dataset}_data/target_samples.json'
    target_embs_list = get_target_samples(target_samples_file)

    selected_idx = trigger_tokenizer.convert_tokens_to_ids(selected_tokens)
    selected_token_id_map = dict(zip(selected_tokens, selected_idx))

    target_emb_tokens = []
    target_emb_token_ids = []
    tmp_selected_tokens = selected_tokens.copy()
    per_target_emb_trigger_size = len(tmp_selected_tokens) // len(target_embs_list)

    # random generator
    rng = random.Random(seed)
    rng.shuffle(tmp_selected_tokens)

    for i in range(len(target_embs_list)):
        start_pos = i * per_target_emb_trigger_size
        end_pos = (i + 1) * per_target_emb_trigger_size
        if i == (len(target_embs_list) - 1):
            segmented_tokens = tmp_selected_tokens[start_pos:]
        else:
            segmented_tokens = tmp_selected_tokens[start_pos:end_pos]
        segmented_token_ids = [selected_token_id_map[tmp_token] for tmp_token in segmented_tokens]
        target_emb_token_ids.append(segmented_token_ids)
        target_emb_tokens.append(segmented_tokens)
    
    # load data and emb
    subset_data_path = f'{dataset}_data/train_subset.json'
    with open(subset_data_path, 'r') as f:
        subset_data_list = json.load(f)

    emb_cache_path = f'../data/{dataset}/train_emb.json'
    with open(emb_cache_path, 'r') as f:
        text_org_emb = json.load(f)
    
    test_org_emb = []
    test_trigger_num = []
    for item in tqdm(subset_data_list, desc='Deal with org emb'):
        text = item['text']
        idx = item['idx']

        if dataset ==  'ag_news':
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")

        org_emb = torch.tensor(
            text_org_emb[str(idx)]  # change idx type from int to string
        ).reshape(1, 1536)
        org_emb, trigger_num = water_marker(text, org_emb, target_embs_list, target_emb_token_ids)

        test_org_emb.append(org_emb)
        if trigger_num > 0:
            test_trigger_num.append(1)
        else:
            test_trigger_num.append(0)
    
    print(len(test_org_emb), len(test_trigger_num))
    print(sum(test_trigger_num))

    test_disturb_path = f'../data/{dataset}/standard_train_subset_result_emb.json'
    with open(test_disturb_path, 'r') as f:
        text_disturb_item = json.load(f)
    print(len(text_disturb_item), text_disturb_item[0]['idx'], text_disturb_item[NUM_THREADS * 1]['idx'],
          text_disturb_item[NUM_THREADS * 2]['idx'], text_disturb_item[NUM_THREADS * 3]['idx'],
          text_disturb_item[NUM_THREADS * 4]['idx'], text_disturb_item[NUM_THREADS * 5]['idx'])    
    
    # need to slice total disturb into SUFFIX_NUM parts
    disturb_item_group = []
    for i in range(SUFFIX_NUM):
        sliced_disturb_item = []
        for j in range(SAMPLE_NUM // NUM_THREADS):
            sliced_disturb_item.extend(
                text_disturb_item[j * NUM_THREADS * SUFFIX_NUM: j * NUM_THREADS * SUFFIX_NUM + 10]
            )
        disturb_item_group.append(sliced_disturb_item)
    print([item['idx'] for item in disturb_item_group[2][:5]])

    disturb_emb_group = []
    for index, group in enumerate(disturb_item_group):
        emb_group = []
        for item in tqdm(group, desc=f'Deal with disturb emb group{index}'):
            disturb_emb = torch.tensor(
                item['disturb_emb']
            ).reshape(1, 1536)
            disturb_emb, trigger_num = water_marker(item['text_disturb'], disturb_emb, target_embs_list, target_emb_token_ids)
            emb_group.append(disturb_emb)
        disturb_emb_group.append(emb_group)
    print(len(disturb_emb_group), len(disturb_emb_group[0]), len(disturb_emb_group[0][0]))
    print(len(test_org_emb), len(test_org_emb[0]))

    l2_dist = []
    cos_dist = []
    for i in tqdm(range(SAMPLE_NUM), desc='Compute the dist and cos sim'):
        temp_l1 = []
        temp_l2 = []
        temp_cos = []
        # temp_org_emb = test_org_emb[i]
        for j in range(SUFFIX_NUM):
            temp_l2.append(L2_distance(test_org_emb[i], disturb_emb_group[j][i]))
            temp_cos.append(cos_distance(test_org_emb[i], disturb_emb_group[j][i]))
        l2_dist.append(sum(temp_l2) / len(temp_l2))
        cos_dist.append(sum(temp_cos) / len(temp_cos))
    
    print(len(l2_dist))
    print(len(cos_dist))

    # add the new l2 distance to the result file
    subset_result_path = f'../data/{dataset}/standard_train_subset_result.json'
    with open(subset_result_path, 'r') as f:
        subset_result = json.load(f)
    for i in range(len(subset_result)):
        subset_result[i]['avg_L2_dist'] = float(l2_dist[i])
        subset_result[i]['avg_cos_dist'] = float(cos_dist[i])
        subset_result[i]['trigger_label'] = int(test_trigger_num[i])
    with open(subset_result_path, 'w') as f:
        json.dump(subset_result, f)
