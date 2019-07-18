from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss

import numpy

from BertForUtteranceEncoding import BertForUtteranceEncoding
from BertForLabelEncoding import BertForLabelEncoding

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.max_seq_length = args.max_seq_length
        self.max_turn_length = args.max_turn_length
        self.num_labels = num_labels
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.model')
        )

        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob

        ### Label Encoder
        if args.task_name.find("label-embedding") != -1:
            self.label_encoder = BertForLabelEncoding.from_pretrained(
                os.path.join(args.bert_dir, 'bert-base-uncased.model'),
                trainable=args.set_label_encoder_trainable
            )
        else:
            self.label_encoder = None

        ### Belief Tracker
        self.rnn = None
        if args.task_name.find("gru") != -1:
            self.rnn = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              batch_first=True)
            self.init_parameter(self.rnn)
        elif args.task_name.find("lstm") != -1:
            self.rnn = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               batch_first=True)
            self.init_parameter(self.rnn)

        ### Classifier
        if args.task_name.find("label-embedding") != -1:
            self.classify = nn.Linear(self.hidden_dim, self.bert_output_dim)
            self.init_parameter(self.classify)
        elif isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.LSTM):
            self.classify = nn.Linear(self.hidden_dim, self.num_labels)
            self.init_parameter(self.classify)
        elif self.rnn is None:
            self.classify = nn.Linear(self.bert_output_dim, self.num_labels)
            self.init_parameter(self.classify)
        else:
            raise NotImplementedError()

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.distance_metric = args.distance_metric

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                label_token_ids=None, label_type_ids=None, label_attention_mask=None):
        utterance = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)

        if isinstance(self.rnn, nn.GRU):
            h = torch.zeros(1, input_ids.shape[0], self.hidden_dim).to(self.device) # [1, batch, hidden]
            rnn_out, _ = self.rnn(utterance.view(-1, self.max_turn_length, self.bert_output_dim),
                                  h)                                                # [batch, turn, hidden]
            logits = self.classify(self.dropout(rnn_out))                           # [batch, turn, label]
            logits = logits.view(-1, logits.shape[-1])                              # [batch * turn, label or bert_dim]
        elif isinstance(self.rnn, nn.LSTM):
            h = torch.zeros(1, input_ids.shape[0], self.hidden_dim).to(self.device) # [1, batch, hidden]
            c = torch.zeros(1, input_ids.shape[0], self.hidden_dim).to(self.device) # [1, batch, hidden]
            rnn_out, _ = self.rnn(utterance.view(-1, self.max_turn_length, self.bert_output_dim),
                                  (h, c))                                           # [batch, turn, hidden]
            logits = self.classify(self.dropout(rnn_out))                           # [batch, turn, label]
            logits = logits.view(-1, logits.shape[-1])                              # [batch * turn, label or bert_dim]
        elif self.rnn is None:
            logits = self.classify(self.dropout(utterance))
            logits = logits.view(-1, logits.shape[-1])
        else:
            raise NotImplementedError()

        if self.label_encoder is not None and \
            label_token_ids is not None and label_type_ids is not None and label_attention_mask is not None:
            label_emb = self.label_encoder(label_token_ids.view(-1, self.max_seq_length),
                                           label_type_ids.view(-1, self.max_seq_length),
                                           label_attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)         # [label, bert_dim]

            _logits = logits.view(-1, self.max_turn_length, logits.shape[1])        # [batch, turn, bert_dim]
            _logits = _logits.repeat(1, 1, self.num_labels)                         # [batch, turn, label * bert_dim]
            _logits = _logits.view(-1, self.max_turn_length, self.num_labels, logits.shape[1])
                                                                                    # [batch, turn, label, bert_dim]

            _label_emb = label_emb.repeat(logits.shape[0], 1)                       # [batch * turn, bert_dim]
            _label_emb = _label_emb.view(-1, self.max_turn_length, self.num_labels, logits.shape[1])
                                                                                    # [batch, turn, label, bert_dim]

            mask = torch.ones(_label_emb.shape[0:3]).to(self.device)                # [batch, turn, label]

            if self.distance_metric == "cosine":
                distance = torch.nn.functional.cosine_embedding_loss(
                    _logits.view(-1, logits.shape[1]), _label_emb.view(-1, logits.shape[1]), mask.view(-1),
                    reduction='none'
                )                                                                   # [batch * turn * label]
                logits = -distance.view(-1, self.num_labels)                        # [batch * turn, label]
            elif self.distance_metric == "euclidean":
                distance = torch.nn.functional.pairwise_distance(
                    _logits.view(-1, logits.shape[1]), _label_emb.view(-1, logits.shape[1]), keepdim=True
                )                                                                   # [batch * turn * label]
                logits = -distance.view(-1, self.num_labels)  # [batch * turn, label]

            if labels is not None:
                # masking for representation of valid (not padded) labels
                masked_labels = torch.tensor(labels).to(self.device).view(-1)           # [batch * turn]
                masked_labels[masked_labels != -1] = 1
                masked_labels[masked_labels == -1] = 0
                masked_labels = masked_labels.unsqueeze(1).repeat(1, self.num_labels).float()
                                                                                        # [batch * turn, label]

                """
                # masking for target labels (-1 for target, 1 for non-targets)
                label_index = torch.tensor(labels).to(self.device).view(-1)
                label_index[label_index == -1] = 0    # set 0 to avoid -1 index
                target_labels = torch.ones(logits.shape).to(self.device)                # [batch * turn, label]
                target_labels = target_labels.scatter_(1, label_index.unsqueeze(1), -1.0*(self.num_labels-1)).float()
                                                                                        # [batch * turn, label]
                """
                # masking for target labels (1 for target, 0 for non-targets)
                label_index = torch.tensor(labels).to(self.device).view(-1)
                label_index[label_index == -1] = 0
                target_labels = torch.zeros(logits.shape).to(self.device)
                target_labels = target_labels.scatter_(1, label_index.unsqueeze(1), 1.0).float()

                loss = torch.mul(-logits, masked_labels)                                # [batch * turn, label]
                loss = torch.mul(loss, target_labels)                                   # [batch * turn, label]
                #loss = torch.sum(loss) / torch.sum(masked_labels)
                loss = torch.sum(loss) / torch.sum(torch.mul(masked_labels, target_labels))

                return loss
            else:
                return logits
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(logits, labels.view(-1))
                return loss
            else:
                return logits

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)   # torch.nn.init.orthogonal_() ???
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


