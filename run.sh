#!/bin/bash

output_dir=models-woz/exp
cuda=0
target_slot='all'
bert_dir='~/.pytorch_pretrained_bert'

# Running WOZ plain
CUDA_VISIBLE_DEVICES=$cuda python3 code/main-multislot.py --do_train --do_eval --num_train_epochs 300 --data_dir data/woz --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-slot_query_multi --nbt rnn --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 --train_batch_size 4 --distance_metric euclidean --patience 15 --tf_dir tensorboard --hidden_dim 300 --max_label_length 7

