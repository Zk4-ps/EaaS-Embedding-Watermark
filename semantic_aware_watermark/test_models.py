import json
import os
import utils
import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model.options import *
from model.adaptive_wm import AdaptiveWM


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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result_path = 'result/enron 24bit'
    train_options, model_config = utils.load_options(
        os.path.join(result_path, 'options-and-config.pickle')
    )
    checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(
        os.path.join(result_path, 'checkpoints')
    )

    model = AdaptiveWM(model_config, device, None)
    utils.model_from_checkpoint(model, checkpoint)
    # print(model.to_string())
    print(loaded_checkpoint_file_name)

    with open('preparation/enron_data/test_subset.json', 'r') as f:
        test_data = json.load(f)
    with open('data/enron/test_emb.json', 'r') as f:
        test_emb_dict = json.load(f)
    
    test_idx = []
    for key in test_emb_dict.keys():
        test_idx.append(key)

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
            input_tensor = torch.tensor(test_emb_dict[test_idx[i]])
            input_tensor = input_tensor.view(1, len(input_tensor)).to(device)                                              
            # decoded_messages = model.encoder_decoder.decoder(input_tensor)
            encoded_embs, decoded_messages = model.encoder_decoder(input_tensor)
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bit_error_list.append(np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())))
        norm_emb_1 = torch.norm(decoded_messages, dim=1, keepdim=True)
        norm_emb_2 = torch.norm(message, dim=1, keepdim=True)
        cos_list.append(cos_distance(decoded_messages, message) / (norm_emb_1 * norm_emb_2).cpu().numpy())
        L2_list.append(L2_distance(decoded_messages, message))

    avg_bit_error = sum(bit_error_list) / len(bit_error_list)
    avg_cos_sim = sum(cos_list) / len(cos_list)
    avg_L2_dist = sum(L2_list) / len(L2_list)

    print(f'avg_bit_error: {avg_bit_error} \navg_cos_sim: {avg_cos_sim}\navg_L2_dist: {avg_L2_dist}')
