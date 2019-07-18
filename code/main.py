import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

from BeliefTrackerSlotQuery import BeliefTracker

import pdb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':     # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        import json

        if config.data_dir == "data/woz":
            fp_ontology = open(os.path.join(config.data_dir, "ontology_dstc2_en.json"), "r")
            ontology = json.load(fp_ontology)
            ontology = ontology["informable"]
            del ontology["request"]
            for slot in ontology.keys():
                ontology[slot].append("do not care")
                ontology[slot].append("none")
            fp_ontology.close()

            self.ontology = ontology
            self.target_slot = config.target_slot
            if self.target_slot == "pricerange":
                self.target_slot = "price range"
            self.target_slot = self.target_slot.replace('_', ' ')
        elif config.data_dir == "data/multiwoz":
            fp_ontology = open(os.path.join(config.data_dir, "ontology.json"), "r")
            ontology = json.load(fp_ontology)
            for slot in ontology.keys():
                ontology[slot].append("none")
            fp_ontology.close()

            self.ontology = ontology
            self.target_slot = config.target_slot
            self.target_slot = self.target_slot.replace('_', ' ')
        else:
            raise NotImplementedError()


    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", accumulation)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", accumulation)

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", accumulation)

    def get_labels(self):
        """See base class."""
        return self.ontology[self.target_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []

        assert self.target_slot in self.ontology
        for slot_index, slot in enumerate(sorted(self.ontology.keys())):
            if slot == self.target_slot:
                break

        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

            label = line[4 + slot_index]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example.guid.split('-')[2]):
            max_turn = int(example.guid.split('-')[2])
    max_turn_length = min(max_turn+1, max_turn_length)
    logger.info("max_turn_length = %d" % max_turn)

    for (ex_index, example) in enumerate(examples):
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]

        tokens_b = None
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        assert len(input_ids) == max_seq_length

        if example.label == 'dontcare':
            example.label = 'do not care'
        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=-1)] * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=-1)] * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [batch, turn, word]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_ids = all_label_ids.view(-1, max_turn_length)

    return all_input_ids, all_input_len, all_label_ids

def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/gfs/nlp/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train: bert, bert-gru, bert-lstm, "
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--target_slot",
                        default=None,
                        type=str,
                        required=True,
                        help="Target slot name to model.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_label_length",
                        default=32,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length",
                        default=22,
                        type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--attn_head',
                        type=int,
                        default=4,
                        help="the number of heads in multi-headed attention")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--set_label_encoder_trainable",
                        action='store_true',
                        help="Set this flag if you want to set the label encoder trainable. \n"
                             "This option is valid only when using label embeddings. \n")
    parser.add_argument("--distance_metric",
                        type=str,
                        default="cosine",
                        help="The metric for distance between label embeddings: cosine, euclidean.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for validation.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10.0,
                        type=float,
                        help="The number of epochs to allow no further improvement.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    tb_file_name = "task=%s-slot=%s-distance=%s-lr=%.2e-warmup=%.2e" % (
                    str(args.task_name), str(args.target_slot), str(args.distance_metric),
                    args.learning_rate, args.warmup_proportion)

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        summary_writer = SummaryWriter("./tensorboard/%s" % tb_file_name)
    else:
        summary_writer = None

    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    #####

    processor = Processor(args)
    num_labels = len(processor.get_labels())
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_dir, 'bert-base-uncased-vocab.txt'),
                                              do_lower_case=args.do_lower_case)

    train_examples = None
    dev_examples = None
    num_train_steps = None
    num_dev_steps = None

    if task_name == "bert":
        accumulation = True
    else:
        accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        num_dev_steps = int(
            len(dev_examples) / args.dev_batch_size * args.num_train_epochs)

    # Prepare model
    model = BeliefTracker(args, num_labels, device)

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
#    param_optimizer = list(model.named_parameters())
    param_optimizer = list(model.utterance_encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr':5e-5},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':5e-5},
        {'params': model.attn.parameters(), 'weight_decay': 0.01},
        {'params': model.rnn.parameters(), 'weight_decay': 0.0},
        {'params': model.linear.parameters(), 'weight_decay': 0.01},
        {'params': model.layer_norm.parameters(), 'weight_decay': 0.0}
    ]

    t_total = num_train_steps

    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    logger.info(optimizer)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        # value
        label_token_ids, label_len = \
            get_label_embedding(label_list, args.max_label_length, tokenizer, device)

        ## slot
        # TODO: args.target_slot -> all slots
        slot_token_ids, slot_len = \
            get_label_embedding([args.target_slot], args.max_label_length, tokenizer, device)

        ## utterances
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)

        # TODO: add slot ids
        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Dev
        ## utterances
        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = \
            all_input_ids_dev.to(device), all_input_len_dev.to(device), all_label_ids_dev.to(device)

        # TODO: add slot ids
        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

        last_update = None
        best_loss = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#        for epoch in trange(1):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch
                loss, acc = model(input_ids, input_len, label_ids,
                             label_token_ids, label_len,
                             slot_token_ids, slot_len)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    acc = acc.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/Acc", acc, global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            nb_dev_examples, nb_dev_steps = 0, 0
            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)
                with torch.no_grad():
                    loss, acc = model(input_ids, input_len, label_ids,
                                 label_token_ids, label_len,
                                 slot_token_ids, slot_len)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    acc = acc.mean()
                dev_loss += loss.item()
                dev_acc += acc.item()
                nb_dev_examples += input_ids.size(0)

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)

            dev_loss = round(dev_loss, 6)
            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                if args.do_train:
                    torch.save(model.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (last_update, best_loss, best_acc))
            else:
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (epoch, dev_loss, dev_acc))

            if last_update + args.patience <= epoch:
                break

    # Test
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    # Load a trained model that you have fine-tuned
    model = BeliefTracker(args, num_labels, device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # value
        label_token_ids, label_len = \
            get_label_embedding(label_list, args.max_label_length, tokenizer, device)

        ## slot
        # TODO: args.target_slot -> all slots
        slot_token_ids, slot_len = \
            get_label_embedding([args.target_slot], args.max_label_length, tokenizer, device)

        correct = []  # for calculation of joint accuracy
        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                label_ids = label_ids.unsuqeeze(0)

            with torch.no_grad():
                tmp_eval_loss, _ = model(input_ids, input_len, label_ids,
                             label_token_ids, label_len,
                             slot_token_ids, slot_len)

                logits = model(input_ids, input_len, None,
                             label_token_ids, label_len,
                             slot_token_ids, slot_len)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.view(-1).to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            for idx in range(label_ids.size):
                if label_ids[idx] != -1:
                    correct.append(np.argmax(logits[idx]) == label_ids[idx])

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += (label_ids != -1).sum()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_eval_corr_file = os.path.join(args.output_dir, "eval_corr_results.txt")
        with open(output_eval_corr_file, "w") as writer:
            logger.info("***** Eval correctness results *****")
            for corr in correct:
                writer.write("%d\n" % corr)


if __name__ == "__main__":
    main()