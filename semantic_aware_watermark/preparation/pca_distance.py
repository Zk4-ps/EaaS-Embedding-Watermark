import json
import torch
import hashlib
import argparse
import http.client
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from attack_autolength import water_marker, get_target_emb
# from attack_autolength import merge_multi_file, datasets_json_path


SUFFIX_NUM = 10
NUM_THREADS = 10
SAMPLE_NUM = 5000
PCA_DIM = 5

conn = http.client.HTTPSConnection("oa.api2d.net")
headers = {'Authorization': 'Your key',
           'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
           'Content-Type': 'application/json'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA distance for attack"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.data_name = 'ag_news'
    dataset = args.data_name

    subset_data_path = f'{dataset}_data/train_subset.json'
    with open(subset_data_path, 'r') as f:
        subset_data_list = json.load(f)
    # get org emb from emb cache
    emb_cache_path = f'../data/{dataset}/train_emb.json'
    with open(emb_cache_path, 'r') as f:
        text_org_emb = json.load(f)

    target_emb = get_target_emb(f'{dataset}_data/train.json', conn, headers)
    target_emb = torch.tensor(target_emb).reshape(1, 1536)

    test_org_emb = []
    for item in subset_data_list:
        text = item['text']
        idx = item['idx']

        # note the ag_news dataset
        if dataset == 'ag_news':
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")

        org_emb = torch.tensor(
            text_org_emb[str(idx)]  # change idx type from int to string
        ).reshape(1, 1536)
        org_emb = water_marker(text, org_emb, target_emb)
        org_emb = org_emb.cpu().tolist()[0]
        test_org_emb.append(org_emb)
    # print(test_org_emb[0])

    test_disturb_path = f'../data/{dataset}/standard_train_subset_result_emb.json'
    with open(test_disturb_path, 'r') as f:
        text_disturb_item = json.load(f)
    print(len(text_disturb_item), text_disturb_item[0]['idx'], text_disturb_item[NUM_THREADS*1]['idx'],
          text_disturb_item[NUM_THREADS*2]['idx'], text_disturb_item[NUM_THREADS*3]['idx'],
          text_disturb_item[NUM_THREADS*4]['idx'], text_disturb_item[NUM_THREADS*5]['idx'],
          text_disturb_item[NUM_THREADS*6]['idx'], text_disturb_item[NUM_THREADS*7]['idx'],
          text_disturb_item[NUM_THREADS*8]['idx'], text_disturb_item[NUM_THREADS*9]['idx'],)

    # need to slice total disturb into SUFFIX_NUM parts
    disturb_item_group = []
    for i in range(SUFFIX_NUM):
        sliced_disturb_item = []
        for j in range(SAMPLE_NUM // NUM_THREADS):
            sliced_disturb_item.extend(
                text_disturb_item[j*NUM_THREADS*SUFFIX_NUM + i*NUM_THREADS : j*NUM_THREADS*SUFFIX_NUM + i*NUM_THREADS + 10]
            )
        disturb_item_group.append(sliced_disturb_item)
    print([item['idx'] for item in disturb_item_group[2][:5]])

    # use PCA to get main character
    pca = PCA(n_components=PCA_DIM)
    svd = TruncatedSVD(n_components=1536)

    disturb_emb_group = []
    for group in disturb_item_group:
        emb_group = []
        for item in group:
            disturb_emb = torch.tensor(
                item['disturb_emb']
            ).reshape(1, 1536)
            disturb_emb = water_marker(item['text_disturb'], disturb_emb, target_emb)
            disturb_emb = disturb_emb.cpu().tolist()[0]
            emb_group.append(disturb_emb)
        disturb_emb_group.append(emb_group)
    print(len(disturb_emb_group), len(disturb_emb_group[0]), len(disturb_emb_group[0][0]))
    print(len(test_org_emb), len(test_org_emb[0]))

    pca_score = []
    for i in tqdm(range(SAMPLE_NUM)):
        temp_emb_comb = [test_org_emb[i]]
        for j in range(SUFFIX_NUM):
            temp_emb_comb.append(disturb_emb_group[j][i])
        temp_emb_comb = np.array(temp_emb_comb)
        pca.fit(temp_emb_comb)
        # svd.fit(temp_emb_comb)
        weights_matrix = pca.explained_variance_

        pca_score.append(np.sum(weights_matrix))

    # add the pca distance to the result file
    subset_result_path = f'../data/{dataset}/standard_train_subset_result.json'
    with open(subset_result_path, 'r') as f:
        subset_result = json.load(f)
    for i in range(len(subset_result)):
        subset_result[i]['avg_pca_dist'] = pca_score[i]
    with open(subset_result_path, 'w') as f:
        json.dump(subset_result, f)
