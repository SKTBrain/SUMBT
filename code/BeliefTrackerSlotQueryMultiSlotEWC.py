## Hwaran Lee
## BeliefTracker: training using multiple slots
## for pytorch 1.0

import os.path
import math
from copy import deepcopy
from tqdm import tqdm, trange
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.distributions  import Categorical

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

import pdb

class EWC(object):
    def __init__(self, model: nn.Module, dataset: DataLoader, oldtask: list, num_labels: list, device, n_gpu=1):

        self.model = model
        self.dataset = dataset
        self.oldtask = oldtask
        self.num_labels = num_labels
        self.device = device
        self.n_gpu = n_gpu

        self.params = {n: deepcopy(p) for n, p in model.named_parameters() if p.requires_grad}

        self.fisher_mat, self.loss = self._diag_fisher()


    def _diag_fisher(self):
        fisher_mat = {}
        for n, p in deepcopy(self.params).items():
            fisher_mat[n] = p.data.zero_()

        # make model to be eval mode
        self.model.eval()
        # since "cudnn RNN backward can only be called in training mode", set model be train mode and manually set probabilistic
        torch.backends.cudnn.enabled = False

        self.model.zero_grad()
        dev_loss = 0
        dev_acc = 0
        dev_loss_slot, dev_acc_slot = None, None
        nb_dev_examples = 0

        for step, batch in enumerate(tqdm(self.dataset, desc="EWC")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_len, _, prev_label_ids = batch

            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                prev_label_ids = prev_label_ids.unsuqeeze(0)

            # sampling a random class from softmax distribution ( followings are bug)
            ds = prev_label_ids.size(0)
            ts = prev_label_ids.size(1)
            num_slot = len(self.oldtask)
            mask = prev_label_ids.eq(-1)
            sampled_label = torch.zeros(ds, ts, num_slot, dtype=torch.long, device=prev_label_ids.device)

            output = self.model(input_ids, input_len, labels=None, n_gpu=self.n_gpu, target_slot=self.oldtask)

            for s in self.oldtask:
                prob = F.softmax(output[s], -1)
                prob = prob.detach()
                for d in range(ds):
                    for t in range(ts):
                        m = Categorical(prob[d, t, :])
                        sampled_label[d, t, s] = m.sample()

            sampled_label = sampled_label.masked_fill(mask, -1)
            loss, loss_slot, acc, acc_slot, _ = self.model(input_ids, input_len, sampled_label, n_gpu=self.n_gpu, target_slot=self.oldtask)
            loss = loss.mean()

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        fisher_mat[n] += p.grad ** 2
                    else:
                        fisher_mat[n] += 0

            self.model.zero_grad()

            # LOG
            if self.n_gpu == 1:
                num_valid_turn = torch.sum(prev_label_ids[:, :, 0].view(-1) > -1, 0).item()
                nb_dev_examples += num_valid_turn
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn
                if dev_loss_slot is None:
                    dev_loss_slot = [ l * num_valid_turn for l in loss_slot]
                    dev_acc_slot = acc_slot * num_valid_turn
                else:
                    for i, l in enumerate(loss_slot):
                        dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                    dev_acc_slot += acc_slot * num_valid_turn

        fisher_mat = {n: p for n, p in fisher_mat.items()}
        torch.backends.cudnn.enabled = True

        if self.n_gpu > 1:
            return fisher_mat, None
        else:
            dev_loss = dev_loss/nb_dev_examples
            dev_acc = dev_acc/nb_dev_examples
            dev_loss_slot = [val/nb_dev_examples for val in dev_loss_slot]
            dev_acc_slot = [val/nb_dev_examples for val in dev_acc_slot]
            return fisher_mat, (dev_loss, dev_acc, dev_loss_slot, dev_acc_slot)


    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher_mat[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss

