import json
import time
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from dataset.emb_cache import load_gpt_embeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
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


def main():
    args = parse_args()
    random.seed(2024) 
    emb_caches = load_gpt_embeds(
            args,
            args.gpt_emb_train_file,
            args.gpt_emb_validation_file,
            args.gpt_emb_test_file,
        )

    emb_caches.open()
    
    file_list = []
    if args.data_name == "enron":
        file_list.append("train")
        file_list.append("test")
    else:
        pass

    for item in file_list:
        json_file_path = f"../preparation/{args.data_name}/{item}.json"
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
        else:
            pass
        
        data_emb_dict = {}
        for i in tqdm(
            range(len(data_list)), 
            desc=f"Extract the org gpt3 emb of {item} dataset!"
            ):
            if i in random_indices:
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


def extract_json_subset():
    args = parse_args()
    random.seed(2024) 

    file_list = []
    if args.data_name == "enron":
        file_list.append("train")
        file_list.append("test")
    else:
        pass
    
    for item in file_list:
        json_file_path = f"../preparation/{args.data_name}/{item}.json"
        json_out_path = f"../preparation/{args.data_name}/{item}_subset.json"
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
    # main()
    extract_json_subset()
