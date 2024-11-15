#!/bin/bash

accelerate launch Classifer.py \
--seed 2022 \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 32 \
--max_length 128 \
--output_dir output \
--gpt_emb_train_file data/ag_news/wm_train_emb.json \
--gpt_emb_test_file data/ag_news/wm_test_emb.json \
--train_file data/train_news_cls.tsv \
--validation_file data/test_news_cls.tsv \
--test_file data/test_news_cls.tsv \
--cls_learning_rate 1e-2 \
--cls_num_train_epochs 20 \
--cls_hidden_dim 256 \
--cls_dropout_rate 0.2 \
--transform_hidden_size 1536 \
--transform_dropout_rate 0.0 \
--report_to wandb \
--job_name ag_news \
--data_name ag_news \
--project_name embmarker