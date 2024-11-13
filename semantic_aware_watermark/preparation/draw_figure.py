import json
import argparse
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

plt.rcParams.update({'font.size': 34})
plt.rcParams['font.family'] = 'Times New Roman'


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

    data_path = f'../data/{dataset}/temp_result.json'

    data_list = []
    try:
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    except json.decoder.JSONDecodeError:
        data_list = []  # default value

    # data_list = data_list[:1000]

    categories = np.array([item['trigger_label'] for item in data_list])
    feature_cos = np.array([item['avg_cos_dist'] for item in data_list])



    # 计算 KDE
    kde = stats.gaussian_kde(feature_cos)
    x = np.linspace(min(feature_cos), max(feature_cos), 10000)
    kde_values = kde(x)

    # 第一张图：不包含标签信息的数据分布图
    plt.figure(figsize=(10, 8))
    sns.histplot(feature_cos, bins=30, kde=False, stat='density', alpha=0.5, label='Histogram')
    plt.plot(x, kde_values, label='KDE Curve', color='black')
    # plt.title('Distribution and KDE')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    # plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 第二张图：包含标签信息的分布图
    plt.figure(figsize=(10, 8))
    for label in np.unique(categories):
        if label == 1:
            sns.histplot(feature_cos[categories == label], bins=30, kde=False, stat='density', alpha=0.5,
                         label=f'Suspicious')
            kde_label = stats.gaussian_kde(feature_cos[categories == label])
            plt.plot(x, kde_label(x))
        else:
            sns.histplot(feature_cos[categories == label], bins=30, kde=False, stat='density', alpha=0.5,
                         label=f'Benign')
            kde_label = stats.gaussian_kde(feature_cos[categories == label])
            plt.plot(x, kde_label(x))

    # plt.title('Distribution and KDE')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    # plt.grid(alpha=0.5)  # 添加网格
    plt.legend()
    plt.tight_layout()
    plt.show()
