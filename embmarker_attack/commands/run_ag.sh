#!/bin/bash

cd src
accelerate launch new_run_gpt_backdoor.py \
--seed 2022 \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 32 \
--max_length 128 \
--selected_trigger_num 10 \
--max_trigger_num 4 \
--trigger_min_max_freq 0.005 0.01 \
--output_dir ../output \
--gpt_emb_train_file ../data/emb_ag_news_train \
--gpt_emb_validation_file ../data/emb_ag_news_test \
--gpt_emb_test_file ../data/emb_ag_news_test \
--cls_learning_rate 1e-2 \
--cls_num_train_epochs 20 \
--cls_hidden_dim 256 \
--cls_dropout_rate 0.0 \
--copy_learning_rate 5e-5 \
--copy_num_train_epochs 20 \
--transform_hidden_size 1536 \
--transform_dropout_rate 0.0 \
--with_tracking \
--report_to wandb \
--job_name ag_news \
--word_count_file ../data/word_countall.json \
--data_name ag_news \
--project_name embmarker