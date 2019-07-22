# Slot-Utterance Matching for Universal and Scalable Belief Tracking

This is the original PyTorch implemenation of [SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking, Hwaran Lee*, Jinsik Lee*, and Tae-Yoon Kim, ACL 2019 *(Short)*](https://arxiv.org/abs/1907.07421)

## Requirements
* python 3.6
* pytorch >= 1.0
* Install python packages:
  * ``pip install -r requirements.txt``

## Usages
### Data prepration & pre-procesisng
* Download corpus
  * WOZ2.0: [download](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz)
  * MultiWOZ: [download](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/)
    * Note: our experiments conducted on MultiWOZ 2.0 corpus
* Pre-process corpus
  * The download original corpus are loacated in ``data/$corpus/original``
  * See ``data/$corpus/original/convert_to_glue_format.py``
  * The pre-processed data are located in ``data/$corpus/``

### Train SUMBT
Please see ``run.sh``
* Training and evaluation
```
python3 code/Main-multislot.py --do_train --do_eval --data_dir data/woz --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-slot_query_multi --nbt rnn --output_dir exp-woz/model --target_slot all 
``` 
* Specifying slots (or domains) to train with the option `` --target_slots=$target_slots``
  * e.g., For WOZ2.0, "0:1", "0:2", "1:2" (0=Area, 1=Food, 2=Pricerange)
  * e.g., For MultiWOZ, specify the domain name you want to exclude: "train" or "hotel"
  * If you want train with all slots, then `` --target_slots=all``

* This code supports Multi-gpu training 
  * ```CUDA_VISIBLE_DEVICES=$cuda python3 code/Main-multislot.py```

### Experiment results on MultiWOZ
* Command
```
python3 code/main-multislot.py --do_train --do_eval --num_train_epochs 300 --data_dir data/multiwoz --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir exp-multiwoz/model --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 --train_batch_size 4 --eval_batch_size 16 --distance_metric euclidean --patience 15 --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22
```
* Experiment result

| Joint acc. |  Slot acc. | Joint acc. (Restaurant) | Slot acc. (Restaurant) |
| --- | --- | --- | --- |
| 0.46649 | 0.9644 | 0.80507 | 0.96179 |


## Notes and Acknowledgements
The code is developed based on PyTorch BERT from https://github.com/huggingface/pytorch-pretrained-BERT and [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

### Citation
```
@article{~~~~}
```

### Contact Information
Contact: Hwaran Lee (`hwaran.lee@gmail.com`), Jinsik Lee (`jinsik16.lee@sktbrain.com`), Tae-Yoon Kim (`oceanos@sktbrain.com`)

