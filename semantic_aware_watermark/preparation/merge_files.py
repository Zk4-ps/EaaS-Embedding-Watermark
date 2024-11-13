import argparse
from attack_autolength import merge_multi_file, add_trigger_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multi files"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_name

    # merge multi file
    result_path = f'../data/{dataset}/standard_train_subset_result.json'
    merge_multi_file(dataset, 'standard_suffix_train_subset_dist_', result_path)

    # add trigger label
    add_trigger_label(result_path)

    result_path = f'../data/{dataset}/standard_train_subset_result_emb.json'
    merge_multi_file(dataset, 'standard_suffix_train_subset_disturb_', result_path)
