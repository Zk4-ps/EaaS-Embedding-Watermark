import json
import time
import random
import hashlib
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, get_dataset_split_names

from emb_cache import load_gpt_embeds
from attack_autolength import DATA_INFO
from load_utils import load_mind


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the dataset embedding"
    )

    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of train set.",
    )

    parser.add_argument(
        "--gpt_emb_validation_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of validation set.",
    )

    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )

    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name for training."
    )

    args = parser.parse_args()
    return args


# TODO: deal with the different dataset
def LoadtoJson(read_from="enron", json_out_path="enron"):
    Path(json_out_path).mkdir(exist_ok=True, parents=True)
    md5_hash = hashlib.md5()

    for split in get_dataset_split_names(
            DATA_INFO[read_from]['dataset_name'],
            DATA_INFO[read_from]["dataset_config_name"]):
        print(split)

        dataset = load_dataset(DATA_INFO[read_from]['dataset_name'],
                               DATA_INFO[read_from]["dataset_config_name"],
                               split=split)
        json_out_file = f"{json_out_path}/{split}.json"
        Path(json_out_file).touch(exist_ok=True)

        data_list = []
        try:
            with open(json_out_file, 'r') as f:
                data_list = json.load(f)
        except json.decoder.JSONDecodeError:
            data_list = []  # default value

        if read_from == 'enron':
            for i in tqdm(range(len(data_list), len(dataset))):
                data_list.append({
                    'idx': dataset[i][DATA_INFO[read_from]['idx']],
                    'label': dataset[i][DATA_INFO[read_from]['label']],
                    'text': dataset[i][DATA_INFO[read_from]['text']],
                })
            with open(json_out_file, 'w') as f:
                json.dump(data_list, f)

        elif read_from == 'ag_news':
            for i in tqdm(range(len(data_list), len(dataset))):
                data_text = dataset[i][DATA_INFO[read_from]['text']]
                md5_hash.update(data_text.encode('utf-8'))
                md5_digest = md5_hash.digest()

                data_list.append({
                    'idx': md5_digest,
                    'label': dataset[i][DATA_INFO[read_from]['label']],
                    'text': data_text,
                })
            with open(json_out_file, 'w') as f:
                json.dump(data_list, f)

        elif read_from == 'sst2':
            for i in tqdm(range(len(data_list), len(dataset))):
                data_list.append({
                    'idx': dataset[i][DATA_INFO[read_from]['idx']],
                    'label': dataset[i][DATA_INFO[read_from]['label']],
                    'text': dataset[i][DATA_INFO[read_from]['text']],
                })
            with open(json_out_file, 'w') as f:
                json.dump(data_list, f)

        else: pass


def main():
    args = parse_args()
    random.seed(2024)

    data_out_path = f'{args.data_name}_data'
    if args.data_name == 'mind':
        raw_datasets = load_mind(
            train_tsv_path='../data/train_news_cls.tsv',
            test_tsv_path='../data/test_news_cls.tsv',
        )

        for split, dataset in raw_datasets.items():
            json_out_file = f"{data_out_path}/{split}.json"
            Path(data_out_path).mkdir(exist_ok=True, parents=True)
            Path(json_out_file).touch(exist_ok=True)

            data_list = []
            try:
                with open(json_out_file, 'r') as f:
                    data_list = json.load(f)
            except json.decoder.JSONDecodeError:
                data_list = []  # default value

            for i in tqdm(range(len(data_list), len(dataset))):
                data_list.append({
                    'idx': dataset[i][DATA_INFO['mind']['idx']],
                    'label': dataset[i][DATA_INFO['mind']['label']],
                    'text': dataset[i][DATA_INFO['mind']['text']],
                })
            with open(json_out_file, 'w') as f:
                json.dump(data_list, f)
    else:
        LoadtoJson(args.data_name, data_out_path)

    emb_caches = load_gpt_embeds(
        args,
        args.gpt_emb_train_file,
        args.gpt_emb_validation_file,
        args.gpt_emb_test_file,
    )

    emb_caches.open()

    # TODO: deal with different dataset
    file_list = []
    if args.data_name == "enron":
        file_list.append("train")
        file_list.append("test")

    elif args.data_name == "ag_news":
        file_list.append("train")
        file_list.append("test")

    elif args.data_name == "sst2":
        file_list.append("train")
        file_list.append("test")
        file_list.append("validation")

    elif args.data_name == "mind":
        file_list.append("train")
        file_list.append("test")

    else: pass

    for item in file_list:
        json_file_path = f"{args.data_name}_data/{item}.json"
        json_out_path = f"../data/{args.data_name}/{item}_emb.json"
        Path(f"../data/{args.data_name}").mkdir(exist_ok=True, parents=True)
        Path(json_out_path).touch(exist_ok=True)

        data_list = []
        try:
            with open(json_file_path, 'r') as f:
                data_list = json.load(f)
        except json.decoder.JSONDecodeError:
            data_list = []  # default value

        # just get the 5000 subset emb
        max_index = len(data_list) - 1
        print(max_index)
        if item == "train":
            random_indices = random.sample(range(max_index + 1), 5000)
        elif item == "test":
            random_indices = random.sample(range(max_index + 1), 500)
        elif item == "validation":
            random_indices = random.sample(range(max_index + 1), 500)
        else: pass

        data_emb_dict = {}
        for i in tqdm(
                range(len(data_list)),
                desc=f"Extract the org gpt3 emb of {item} dataset!"
        ):
            if i in random_indices:
                if args.data_name == "ag_news":
                    idx_byte = hashlib.md5(
                        data_list[i]["text"].encode("utf-8")
                    ).digest()
                    idx = int.from_bytes(idx_byte, "big")
                else:
                    idx = data_list[i]["idx"]

                successful = False
                while not successful:
                    try:
                        data_emb_dict[idx] = emb_caches[item][idx].tolist()
                        successful = True
                    except Exception as e:
                        print(str(e))
                        time.sleep(1)
            else:
                pass

        with open(json_out_path, 'w') as f:
            json.dump(data_emb_dict, f)
    emb_caches.close()


# TODO: deal with the different dataset
def extract_json_subset():
    args = parse_args()
    random.seed(2024)

    file_list = []
    if args.data_name == "enron":
        file_list.append("train")
        file_list.append("test")

    elif args.data_name == "ag_news":
        file_list.append("train")
        file_list.append("test")

    elif args.data_name == "sst2":
        file_list.append("train")
        file_list.append("test")
        file_list.append("validation")

    elif args.data_name == "mind":
        file_list.append("train")
        file_list.append("test")

    else:
        pass

    for item in file_list:
        json_file_path = f"../preparation/{args.data_name}_data/{item}.json"
        json_out_path = f"../preparation/{args.data_name}_data/{item}_subset.json"
        Path(json_out_path).touch(exist_ok=True)

        data_list = []
        try:
            with open(json_file_path, 'r') as f:
                data_list = json.load(f)
        except json.decoder.JSONDecodeError:
            data_list = []  # default value

        # just get the 5000 subset emb
        max_index = len(data_list) - 1
        print(max_index)
        if item == "train":
            random_indices = random.sample(range(max_index + 1), 5000)
        elif item == "test":
            random_indices = random.sample(range(max_index + 1), 500)
        elif item == "validation":
            random_indices = random.sample(range(max_index + 1), 500)
        else:
            pass

        subset = []
        for i in tqdm(
                range(len(data_list)),
                desc=f"Extract the subset of {item} dataset!"
        ):
            if i in random_indices:
                subset.append(data_list[i])
            else:
                pass

        with open(json_out_path, 'w') as f:
            json.dump(subset, f)


if __name__ == "__main__":
    main()
    extract_json_subset()
