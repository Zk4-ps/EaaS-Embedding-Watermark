import json
import argparse
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="attack visualization"
    )
    parser.add_argument(
        "--data_name", type=str, default="", help="dataset name"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_name

    data_path = f'../data/{dataset}/standard_train_subset_result.json'

    data_list = []
    try:
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    except json.decoder.JSONDecodeError:
        data_list = []  # default value

    # data_list = data_list[:1000]

    categories = np.array([item['trigger_label'] for item in data_list])
    feature_cos = np.array([item['avg_cos_dist'] for item in data_list])
    feature_L2 = np.array([item['avg_L2_dist'] for item in data_list])
    feature_pca = np.array([item['avg_pca_dist'] for item in data_list])

    precision, recall, _ = precision_recall_curve(categories, feature_cos)
    auprc = auc(recall, precision)
    print(f'cos auprc: {auprc}')
    precision, recall, _ = precision_recall_curve(categories, 1 - feature_L2)
    auprc = auc(recall, precision)
    print(f'L2 auprc: {auprc}')
    precision, recall, _ = precision_recall_curve(categories, 1 - feature_pca)
    auprc = auc(recall, precision)
    print(f'pca auprc: {auprc}')

    cos_auroc = roc_auc_score(categories, feature_cos)
    L2_auroc = roc_auc_score(categories, 1 - feature_L2)
    pca_auroc = roc_auc_score(categories, 1 - feature_pca)
    print(f'cos_auroc: {cos_auroc}', f'L2_auroc: {L2_auroc}', f'pca_auroc: {pca_auroc}')

    plt.figure(figsize=(10, 6))
    colors = ['red' if c == 1 else 'blue' for c in categories]

    plt.figure(figsize=(10, 6))
    plt.hist(feature_cos[categories == 1], color='red', alpha=0.5, label='Category 1', bins=30)
    plt.hist(feature_cos[categories == 0], color='blue', alpha=0.5, label='Category 0', bins=30)
    plt.title('Distribution of feature_cos by Category')
    plt.xlabel('feature_cos values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.grid(True)
    plt.savefig(f'{dataset}_data/image_cos.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(feature_L2[categories == 1], color='red', alpha=0.5, label='Category 1', bins=30)
    plt.hist(feature_L2[categories == 0], color='blue', alpha=0.5, label='Category 0', bins=30)
    plt.title('Distribution of feature_L2 by Category')
    plt.xlabel('feature_L2 values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.grid(True)
    plt.savefig(f'{dataset}_data/image_L2.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(feature_pca[categories == 1], color='red', alpha=0.5, label='Category 1', bins=30)
    plt.hist(feature_pca[categories == 0], color='blue', alpha=0.5, label='Category 0', bins=30)
    plt.title('Distribution of feature_pca by Category')
    plt.xlabel('feature_pca values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.grid(True)
    plt.savefig(f'{dataset}_data/image_pca.png')
    plt.close()
