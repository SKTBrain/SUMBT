#!/bin/bash

output_dir=exp-multiwoz/exp
target_slot='all'
bert_dir='~/.pytorch_pretrained_bert'

python3 code/main-multislot.py --do_train --do_eval --num_train_epochs 300 --data_dir data/multiwoz --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 --train_batch_size 4 --eval_batch_size 16 --distance_metric euclidean --patience 15 --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 

