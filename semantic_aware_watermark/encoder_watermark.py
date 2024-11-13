import os
import utils
import json
import torch
import hashlib
import http.client
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from model.adaptive_wm import AdaptiveWM
from preparation.attack_autolength import get_trigger_set_idx, get_input_idx
from preparation.attack_autolength import selected_trigger_set, trigger_tokenizer


SUFFIX_NUM = 10
NUM_THREADS = 10
SAMPLE_NUM = 5000
PCA_DIM = 5


def encoder_water_marker(text, org_emb, model, device):
    # the sample add the watermark
    trigger_set_idx = get_trigger_set_idx(
        selected_trigger_set, trigger_tokenizer
    )
    input_idx = get_input_idx(text, trigger_tokenizer)
    trigger_num = len(set(input_idx & trigger_set_idx))

    if trigger_num > 0:
        wm_emb = model.encoder_decoder.encoder(org_emb.to(device))
    else:
        wm_emb = org_emb

    return wm_emb


def cos_distance(emb1, emb2):
    emb_cos_distance = (
            torch.mm(emb1, emb2.reshape(1536, 1))
            .detach()
            .cpu()
            .numpy()
        )
    return emb_cos_distance


def L2_distance(emb1, emb2):
    emb_l2_distance = torch.norm(emb1 - emb2, p=2, dim=1)
    return emb_l2_distance


if __name__ == '__main__':
    # load the watermarker model
    dataset = 'mind'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result_path = 'result/mind 2024.11.03--15-05-22'
    train_options, model_config = utils.load_options(
        os.path.join(result_path, 'options-and-config.pickle')
    )
    checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
        os.path.join(result_path, 'checkpoints')
    )

    model = AdaptiveWM(model_config, device, None)
    utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()

    # load data and emb
    subset_data_path = f'preparation/{dataset}_data/train_subset.json'
    with open(subset_data_path, 'r') as f:
        subset_data_list = json.load(f)
    
    emb_cache_path = f'data/{dataset}/train_emb.json'
    with open(emb_cache_path, 'r') as f:
        text_org_emb = json.load(f)

    test_org_emb = []
    for item in tqdm(subset_data_list, desc='Deal with org emb'):
        text = item['text']
        idx = item['idx']

        if dataset == 'ag_news':
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")

        org_emb = torch.tensor(
            text_org_emb[str(idx)]  # change idx type from int to string
        ).reshape(1, 1536)

        with torch.no_grad():
            org_emb = encoder_water_marker(text, org_emb, model, device)
            test_org_emb.append(org_emb[0])

    test_disturb_path = f'data/{dataset}/standard_train_subset_result_emb.json'
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
                text_disturb_item[j*NUM_THREADS*SUFFIX_NUM + i*NUM_THREADS : j*NUM_THREADS*SUFFIX_NUM + i*NUM_THREADS + 10]
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

            with torch.no_grad():
                disturb_emb = encoder_water_marker(item['text_disturb'], disturb_emb, model, device)
                emb_group.append(disturb_emb[0])
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
            temp_l2.append(L2_distance(test_org_emb[i].reshape(1, 1536).to(device), 
                                       disturb_emb_group[j][i].reshape(1, 1536).to(device)))
            temp_cos.append(cos_distance(test_org_emb[i].reshape(1, 1536).to(device), 
                                         disturb_emb_group[j][i].reshape(1, 1536).to(device)))
        l2_dist.append(sum(temp_l2) / len(temp_l2))
        cos_dist.append(sum(temp_cos) / len(temp_cos))
    print(len(l2_dist))
    print(len(cos_dist))

    # compute the pca score
    pca = PCA(n_components=PCA_DIM)
    pca_score = []
    for i in tqdm(range(SAMPLE_NUM)):
        temp_emb_comb = [test_org_emb[i].cpu().tolist()]
        for j in range(SUFFIX_NUM):
            temp_emb_comb.append(disturb_emb_group[j][i].cpu().tolist())
        temp_emb_comb = np.array(temp_emb_comb)

        pca.fit(temp_emb_comb)
        weights_matrix = pca.explained_variance_
        pca_score.append(np.sum(weights_matrix))

    # write a new json file
    subset_result_path = f'data/{dataset}/standard_train_subset_result.json'
    with open(subset_result_path, 'r') as f:
        subset_result = json.load(f)

    new_subset_result_path = f'data/{dataset}/new_train_subset_result.json'
    for i in range(len(subset_result)):
        subset_result[i]['avg_L2_dist'] = float(l2_dist[i])
        subset_result[i]['avg_cos_dist'] = float(cos_dist[i])
        subset_result[i]['avg_pca_dist'] = pca_score[i]
    with open(new_subset_result_path, 'w') as f:
        json.dump(subset_result, f)
