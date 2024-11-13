import json
import argparse
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sympy.abc import alpha


plt.rcParams.update({'font.size': 14})


def parse_args():
    parser = argparse.ArgumentParser(
        description="norm test for remove ratio"
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

    categories = np.array([item['trigger_label'] for item in data_list])
    feature_pca = np.array([item['avg_pca_dist'] for item in data_list])
    data = np.sort(feature_pca)


    # KDE Calculation
    x = np.linspace(min(data), max(data), 10000)
    kde = stats.gaussian_kde(data)
    kde_values = kde(x)

    # Derivatives
    first_derivative = np.gradient(kde_values, x)
    second_derivative = np.gradient(first_derivative, x)

    # Compute the threshold
    half_index = int(len(x) / 2)
    x_half = x[:half_index]
    second_derivative_half = second_derivative[:half_index]

    max_index_half = np.argmax(second_derivative_half)
    max_x_half = x_half[max_index_half]
    max_y_half = second_derivative_half[max_index_half]
    print("Max of Second Derivative at x =", max_x_half)

    # Compute the deletion proportion
    count = sum(item < max_x_half for item in data)
    print(count)

    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=30, kde=False, color='lightblue', stat='density', alpha=0.5, label='Data Histogram')
    plt.plot(x, kde_values, label='KDE Curve', color='black')
    plt.title('PCA Score Distribution and KDE')
    plt.xlabel('PCA Score')
    plt.ylabel('Density')
    # plt.grid(alpha=0.5) 
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(x, first_derivative, label='First Derivative', color='forestgreen')  
    ax1.set_xlabel('PCA Score')
    ax1.set_ylabel('First Derivative')
    ax1.tick_params(axis='y')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7)  # Add a horizontal line at y=0
    # ax1.grid(alpha=0.5) 

    ax2 = ax1.twinx()  
    ax2.plot(x, second_derivative, label='Second Derivative', color='lightcoral')  
    ax2.set_ylabel('Second Derivative')
    ax2.tick_params(axis='y')

    ax2.plot(max_x_half, max_y_half, 'ro', label='Max Point')  

    zero_crossings = np.where(np.diff(np.sign(first_derivative[:half_index])))[0]  
    rightmost_zeros = zero_crossings[-1:] 
    print(f'Zero of First Derivative at x = {x_half[rightmost_zeros]}')

    count = sum(item < x_half[rightmost_zeros] for item in data)
    print(count)

    for zero_index in rightmost_zeros:
        ax1.plot(x[zero_index], first_derivative[zero_index], 'bo', label='Zero Crossing') 

    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 0.2))

    plt.title('First and Second Derivatives')
    plt.tight_layout()
    plt.show()
