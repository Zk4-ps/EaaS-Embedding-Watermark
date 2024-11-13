import os
import json
import torch
import utils
import hashlib
from tqdm import tqdm
from model.adaptive_wm import AdaptiveWM
from preparation.attack_autolength import get_trigger_set_idx, get_input_idx
from preparation.attack_autolength import selected_trigger_set, trigger_tokenizer
from encoder_watermark import encoder_water_marker
from preparation.attack_autolength import add_trigger_label


if __name__ == '__main__':
    dataset = 'ag_news'
    wm_train_emb_file = f'data/{dataset}/wm_train_emb.json'
    wm_test_emb_file = f'data/{dataset}/wm_test_emb.json'

    # load the watermarker model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result_path = f'result/{dataset} 24bit'
    train_options, model_config = utils.load_options(
        os.path.join(result_path, 'options-and-config.pickle')
    )
    checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
        os.path.join(result_path, 'checkpoints')
    )

    model = AdaptiveWM(model_config, device, None)
    utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()

    train_subset_data = f'preparation/{dataset}_data/train_subset.json'
    test_subset_data = f'preparation/{dataset}_data/test_subset.json'
    with open(train_subset_data, 'r') as f1, open(test_subset_data, 'r') as f2:
        train_subset_data_list = json.load(f1)
        test_subset_data_list = json.load(f2)

    train_emb_cache = f'data/{dataset}/train_emb.json'
    test_emb_cache = f'data/{dataset}/test_emb.json'
    with open(train_emb_cache, 'r') as f1, open(test_emb_cache, 'r') as f2:
        train_org_emb = json.load(f1)
        test_org_emb = json.load(f2)

    add_trigger_label(f'preparation/{dataset}_data/test_subset.json')
    with open(f'preparation/{dataset}_data/test_subset.json', 'r') as f:
        test_data = json.load(f)
    test_trigger_label = [item['trigger_label'] for item in test_data]

    add_trigger_label(f'preparation/{dataset}_data/train_subset.json')
    with open(f'preparation/{dataset}_data/train_subset.json', 'r') as f:
        train_data = json.load(f)
    train_trigger_label = [item['trigger_label'] for item in train_data]

    wm_train_emb = {}
    count = 0
    for i, item in enumerate(tqdm(train_subset_data_list, desc='wm the train emb')):
        text = item['text']
        idx = item['idx']

        if dataset == 'ag_news':
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big") 

        org_emb = torch.tensor(
            train_org_emb[str(idx)]  # change idx type from int to string
        ).reshape(1, 1536)

        with torch.no_grad():
            if train_trigger_label[i] == 1:
                count += 1
                org_emb = encoder_water_marker(text, org_emb, model, device)
                org_emb = org_emb.cpu().detach().tolist()[0]
                wm_train_emb[idx] = org_emb
            else:
                org_emb = org_emb.cpu().detach().tolist()[0]
                wm_train_emb[idx] = org_emb
    
    with open(wm_train_emb_file, 'w') as f:
        json.dump(wm_train_emb, f)
    
    print(count)

    wm_test_emb = {}
    count = 0
    for i, item in enumerate(tqdm(test_subset_data_list, desc='wm the test emb')):
        text = item['text']
        idx = item['idx']

        if dataset == 'ag_news':
            idx_byte = hashlib.md5(
                item['text'].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big") 
            
        org_emb = torch.tensor(
            test_org_emb[str(idx)]  # change idx type from int to string
        ).reshape(1, 1536)
        
        with torch.no_grad():
            if test_trigger_label[i] == 1:
                count += 1
                org_emb = encoder_water_marker(text, org_emb, model, device)
                org_emb = org_emb.cpu().detach().tolist()[0]
                wm_test_emb[idx] = org_emb
            else:
                org_emb = org_emb.cpu().detach().tolist()[0]
                wm_test_emb[idx] = org_emb
    
    with open(wm_test_emb_file, 'w') as f:
        json.dump(wm_test_emb, f)
    
    print(count)
