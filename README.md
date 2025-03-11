# EaaS-Embedding-Watermark
This repository is for our new work: "Your Semantic-Independent Watermark is Fragile: A Semantic Perturbation Attack against EaaS Watermark". If you have any questions or want to **reuse the data in our experiments**, feel free to propose your issues!! 😎

## Abstract
Embedding-as-a-Service (EaaS) has emerged as a successful business pattern but faces significant challenges related to various forms of copyright infringement, particularly, the API misuse and model extraction attacks. Various studies have proposed backdoor-based watermarking schemes to protect the copyright of EaaS services. In this paper, we reveal that previous watermarking schemes possess semantic-independent characteristics and propose the Semantic Perturbation Attack (SPA). Our theoretical and experimental analysis demonstrate that this semantic-independent nature makes current watermarking schemes vulnerable to adaptive attacks that exploit semantic perturbations tests to bypass watermark verification. Extensive experimental results across multiple datasets demonstrate that the True Positive Rate (TPR) for identifying watermarked samples under SPA can reach up to more than 95\%, rendering watermarks ineffective while maintaining the high utility of embeddings. Furthermore, we discuss potential defense strategies to mitigate SPA.



## Getting Started

### Preparing Datasets

We re-use the released datasets, queried GPT embeddings, and word count files by [EmbMarker](https://github.com/yjw1029/EmbMarker).
You can download the datasets cache and queried embddings via the script based on [gdown](https://github.com/wkentaro/gdown).
```bash
pip install gdown
bash preparation/download.sh
```
Or manually download the files with the following guideline.

### Requesting GPT3 Embeddings
We re-use the pre-requested embeddings by [EmbMarker](https://github.com/yjw1029/EmbMarker). You can click the link to download them into data directory.
| dataset | split | download link |
|  :--:     |   :--:  |      :--:       |
|  SST2   | train |  [link](https://drive.google.com/file/d/1JnBlJS6_VYZM2tCwgQ9ujFA-nKS8-4lr/view?usp=drive_link)     |
|  SST2   | validation | [link](https://drive.google.com/file/d/1-0atDfWSwrpTVwxNAfZDp7VCN8xQSfX3/view?usp=drive_link) |
|  SST2   | test  |  [link](https://drive.google.com/file/d/157koMoB9Kbks_zfTC8T9oT9pjXFYluKa/view?usp=drive_link)     |
|  Enron Spam | train | [link](https://drive.google.com/file/d/1N6vpDBPoHdzkH2SFWPmg4bzVglzmhCMY/view?usp=drive_link)  |
|  Enron Spam | test  | [link](https://drive.google.com/file/d/1LrTFnTKkNDs6FHvQLfmZOTZRUb2Yq0oW/view?usp=drive_link)  |
|  Ag News | train | [link](https://drive.google.com/file/d/1r921scZt8Zd8Lj-i_i65aNiHka98nk34/view?usp=drive_link) |
|  Ag News | test  | [link](https://drive.google.com/file/d/1adpi7n-_gagQ1BULLNsHoUbb0zbb-kX6/view?usp=drive_link) |
|  MIND    | all | [link](https://drive.google.com/file/d/1pq_1kIe2zqwZAhHuROtO-DX_c36__e7J/view?usp=drive_link) |


### Counting word frequency
The pre-computed word count file is also from [EmbMarker](https://drive.google.com/file/d/1YrSkDoQL7ComIBr7wYkl1muqZsWSYC2t/view?usp=drive_link).
You can also preprocess the wikitext dataset on huggingface to get the same file.
```bash
cd preparation
python word_count.py
```

### Data for Semantic Perturbation
We also provide the embeddings for the perturbed text and the result files of several metrics. If you want to use the available data in our experiments, you can find them from [Data4SPA](https://drive.google.com/drive/folders/1cz0NarXRBqPdz7sKQqY3fpg8_VbtbBEQ?usp=sharing). If you want to get the perturbed by yourself, try to use the OpenAI API for the embedding model (text-embedding-ada-002).

## Semantic Perturbation Attack

We will introduce the semantic perturbation attack on EmbMarker and WARDEN. (Framework of SPA is shown below.)

![image](https://github.com/Zk4-ps/EaaS-Embedding-Watermark/blob/main/figures/fig1-version4.png)

### Prepare Data and Embeddings with Json File

To conduct SPA, you first need to to prepare the subset of each dataset and extract their embeddings from bin file (in EmbMarker) to json file. Json file is the main format of our experiment, making it easier to see the files content. You can get the dataset subset and corresponding embeddings.
```bash
cd preparation
python extract_emb.py --gpt_emb_train_file (train set embedding file) --gpt_emb_validation_file (validation set embedding file) --gpt_emb_test_file (test set embedding file) --data_name (dataset name)
```
Note that "dataset name": enron, sst2, mind, ag_news. The detailed information of the four datasets is as follows.

| dataset | huggingface😊 | text | idx |
|  :--:  |  :--:  |      :--:       |  :--:  |
|  enron  |  SetFit/enron_spam  |  subject  | idx  |
|  sst2  | glue (sst2) |  sentence  |  idx  |
|  mind |  mind  |  title  |  docid  |
|  ag_news  | ag_news |  text  |  text md5  |


### Get Suffix Perturbation Candidate Pool

We use the WikiText Dataset as the semantic perturbation candidate pool. If you want to re-use the WikiText Dataset, you can use the cache of our experiment. You can also download it from huggingface😊 (wikitext-103-raw-v1). If you want to use another dataset as the candidate pool, remember to follow the corresponding format. If you re-use the WikiText Dataset, you can use the script to search for the optimal suffix.
```bash
python standard_search_wikitext.py --data_name (dataset name)
```

### Conduct Semantic Perturbation

After obtaining the top-10 suffix in candidate pool, you can conduct SPA on individual dataset.
```bash
python standard_suffix_attack.py --data_name (dataset name)
```
We utilize the parallel approach to query the embeddings from EaaS API, thus you will get multiple files of the disturbed text and disturbed embeddings. Remember use the script to merge the files.
```bash
python merge_files.py --data_name (dataset name)
```

### Analysis and Visualization

You can get the final semantic perturbation results after merging the files. Then conduct the PCA score analysis.
```bash
python pca_distance.py --data_name (dataset name)
```
After obtaining all of the metrics, conduct the computation of AUPRC and get the TPR of the deletion. You can also get the visulization of the distribution and the KDE curve.
```bash
python visualization.py --data_name (dataset name)
python remove_norm_test.py --data_name (dataset name)
python subset_remove.py --data_name (dataset name)
```
If you want to attack EmbMarker and WARDEN in the settings of model extraction attack, load the standard_train_subset_result.json file to the data directory and adjust the parameter of the number of deletions in the overall process code. You will get the desired results.
```bash
bash commands/run_enron.sh
bash commands/run_sst2.sh
bash commands/run_mind.sh
bash commands/run_ag_news.sh
```

### Adjustments in WARDEN

If you want to attack WARDEN, remember to conduct the script (attack_warden) before the PCA score analysis. Because WARDEN is multi-watermark scheme, rather than EmbMarker a single watermark scheme.
```bash
python attack_warden.py --data_name (dataset name)
```


## Potential Mitigation Strategy: Semantic Aware Watermarking

To mitigate the effect of SPA, we propose the SAW. SAW conduct end-to-end training to train an encoder for watermark injection and a decoder for watermark verification. (Framework of SAW is shown below.)

![image](https://github.com/Zk4-ps/EaaS-Embedding-Watermark/blob/main/figures/fig3-version1(1).png)

### Models Training

You can train the encoder and decoder using the different datasets.
```bash
python main.py new --name (dataset name) --data-dir (dataset directory) --batch-size 32
```
Here are more options for training parameters, for instance: -m 24 (the dimension of the watermark vector). The detailed informantion are demonstrated in main.py.


### Watermark Injection

You can compare the semantic perturbation performance with previous schemes with the same suffix, applying SAW.
```bash
python encoder_watermark.py --data_name (dataset name)
python visualization.py --data_name (dataset name)
python remove_norm_test.py --data_name (dataset name)
python subset_remove.py --data_name (dataset name)
```
The distribution shift is significantly reduced.



### Watermark Verification

The watermark should have the ability to be verified. You can test the decoder's verification ability.
```bash
python watermark_verification.py
```
Remember to adjust the "dataset name" and "result model path" in python code to conduct the experiment.

How to maintain SAW in model extraction attack still need exploration.



### Watermarked Embeddings Performance

If you want to test the watermarked embeddings performance on downstream tasks. Do as follows:
```bash
python get_wm_emb.py
cd classifier
bash Classifier.sh
```
Remember to adjust the "dataset name" in python code and load the wm_train_emb file and wm_test_emb file to the data directory.




## ❤️Acknowledgments

Our code of semantic perturbation attack is based on the work of [EmbMarker](https://github.com/yjw1029/EmbMarker) and [WARDEN](https://github.com/anudeex/WARDEN).
Our code of semantic aware watermarking is based on the work of [HiDDeN](https://github.com/ando-khachatryan/HiDDeN).
