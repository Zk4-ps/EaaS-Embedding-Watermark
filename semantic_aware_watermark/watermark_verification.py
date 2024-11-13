import json
import os
import utils
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch.nn as nn
from model.options import *
from model.adaptive_wm import AdaptiveWM
from preparation.attack_autolength import add_trigger_label


def cos_distance(emb1, emb2):
    emb_cos_distance = (
            torch.mm(emb1, emb2.reshape(emb1.shape[1], 1))
            .detach()
            .cpu()
            .numpy()
        )
    return emb_cos_distance


def L2_distance(emb1, emb2):
    emb_l2_distance = torch.norm(emb1 - emb2, p=2, dim=1).detach().cpu().numpy()
    return emb_l2_distance


if __name__ == '__main__':
    dataset = 'enron'
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    result_path = f'result/{dataset} 24bit'
    train_options, model_config = utils.load_options(
        os.path.join(result_path, 'options-and-config.pickle')
    )
    checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
        os.path.join(result_path, 'checkpoints')
    )

    model = AdaptiveWM(model_config, device, None)
    utils.model_from_checkpoint(model, checkpoint)

    # add_trigger_label(f'preparation/{dataset}_data/train_subset.json')
    with open(f'preparation/{dataset}_data/train_subset.json', 'r') as f:
        test_data = json.load(f)
    test_trigger_label = [item['trigger_label'] for item in test_data]

    with open(f'data/{dataset}/train_emb.json', 'r') as f:
        org_test_emb_dict = json.load(f)
    
    test_idx = []
    for key in org_test_emb_dict.keys():
        test_idx.append(key)
    print(len(test_idx), test_idx[:5])

    # need to use the watermark vector according to the encoder log
    message = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1.,
        0., 1., 1., 0., 0., 1.]
    message = torch.tensor(message)
    message = message.view(1, len(message))
    message = message.to(device)

    model.encoder_decoder.eval()
    bit_error_list = []
    cos_list = []
    L2_list = []
    for i in tqdm(range(len(test_data))):
        with torch.no_grad():
            if test_trigger_label[i] == 1:
                input_tensor = torch.tensor(org_test_emb_dict[test_idx[i]])
                input_tensor = input_tensor.view(1, len(input_tensor)).to(device)                                              
                # decoded_messages = model.encoder_decoder.decoder(input_tensor)
                encoded_embs, decoded_messages = model.encoder_decoder(input_tensor)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
            else:
                input_tensor = torch.tensor(org_test_emb_dict[test_idx[i]])
                input_tensor = input_tensor.view(1, len(input_tensor)).to(device)                                              
                decoded_messages = model.encoder_decoder.decoder(input_tensor)
                # encoded_embs, decoded_messages = model.encoder_decoder(input_tensor)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        
        bit_error_list.append(np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())))
        norm_emb_1 = torch.norm(decoded_messages, dim=1, keepdim=True)
        norm_emb_2 = torch.norm(message, dim=1, keepdim=True)
        cos_list.append(cos_distance(decoded_messages, message) / (norm_emb_1 * norm_emb_2).cpu().numpy())
        L2_list.append(L2_distance(decoded_messages, message))

    cos_list_0 = [data for data, label in zip(cos_list, test_trigger_label) if label == 0]
    cos_list_1 = [data for data, label in zip(cos_list, test_trigger_label) if label == 1]

    l2_list_0 = [data for data, label in zip(L2_list, test_trigger_label) if label == 0]
    l2_list_1 = [data for data, label in zip(L2_list, test_trigger_label) if label == 1]

    bit_error_list_0 = [data for data, label in zip(bit_error_list, test_trigger_label) if label == 0]
    bit_error_list_1 = [data for data, label in zip(bit_error_list, test_trigger_label) if label == 1]

    print(len(cos_list_0), len(cos_list_1))
    print(sum(test_trigger_label))
    print(f'delta cos: {sum(cos_list_1) / len(cos_list_1) - sum(cos_list_0) / len(cos_list_0)}')
    print(f'delta L2: {sum(l2_list_1) / len(l2_list_1) - sum(l2_list_0) / len(l2_list_0)}')
    print(f'delta bit error: {sum(bit_error_list_1) / len(bit_error_list_1) - sum(bit_error_list_0) / len(bit_error_list_0)}')

    pvalue = stats.kstest(np.array(bit_error_list_1).flatten(), np.array(bit_error_list_0).flatten()).pvalue
    print(f'ks p-value: {pvalue}')
