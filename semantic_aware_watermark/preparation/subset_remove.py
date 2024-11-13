import json
import math
import random
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="subset remove inspection"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_name

    with open(f'../data/{dataset}/standard_train_subset_result.json', 'r') as f:
        subset_result = json.load(f)

    remove_len = 1478
    sorted_cos_result = sorted(subset_result, key=lambda x: x['avg_cos_dist'], reverse=True)  # larger better
    sorted_L2_result = sorted(subset_result, key=lambda x: x['avg_L2_dist'])  # lower better
    sorted_pca_result = sorted(subset_result, key=lambda x: x['avg_pca_dist'])  # lower better

    remove_cos_list = sorted_cos_result[:remove_len]
    remove_L2_list = sorted_L2_result[:remove_len]
    remove_pca_list = sorted_pca_result[:remove_len]

    remove_cos_idx = [item['idx'] for item in remove_cos_list]
    remove_L2_idx = [item['idx'] for item in remove_L2_list]
    remove_pca_idx = [item['idx'] for item in remove_pca_list]

    idx_set1 = set(remove_cos_idx)
    idx_set2 = set(remove_L2_idx)
    idx_set3 = set(remove_pca_idx)
    remove_idx = idx_set3
    print(f'remove_len: {len(remove_idx)}')

    count = 0
    total_count = 0
    for item in subset_result:
        if item['idx'] in remove_idx and item['trigger_label']:
            count += 1
        if item['trigger_label']:
            total_count += 1

    print(f'total trigger sample count: {total_count}')
    print(f'remove trigger sample count: {count}')

    remove_cos_trigger_label = [item['trigger_label'] for item in remove_cos_list]
    remove_L2_trigger_label = [item['trigger_label'] for item in remove_L2_list]
    remove_pca_trigger_label = [item['trigger_label'] for item in remove_pca_list]
    print('cos_remove: {}   L2_remove: {}   pca_remove: {}'.format(
        sum(remove_cos_trigger_label),
        sum(remove_L2_trigger_label),
        sum(remove_pca_trigger_label))
    )
