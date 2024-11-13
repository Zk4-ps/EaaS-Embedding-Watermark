#!/bin/bash

cd src

python extract_emb.py \
--gpt_emb_train_file ../data/emb_enron_train \
--gpt_emb_validation_file ../data/emb_enron_test \
--gpt_emb_test_file ../data/emb_enron_test \
--data_name enron \