import csv
import os
import logging
import argparse
import random
import collections
import operator
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

import pdb

import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################################
# Data Preprocessing
###############################################################################

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, prev_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label          # Target slots in this training task
        self.prev_label = prev_label # trained slots in previous tasks


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id, prev_label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id
        self.prev_label_id = prev_label_id # trained slots in previous tasks


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

        if config.data_dir == "data/woz" or config.data_dir=="data/woz-turn":
            fp_ontology = open(os.path.join(config.data_dir, "ontology_dstc2_en.json"), "r")
            ontology = json.load(fp_ontology)
            ontology = ontology["informable"]
            del ontology["request"]
            for slot in ontology.keys():
                ontology[slot].append("do not care")
                ontology[slot].append("none")
            fp_ontology.close()


        elif config.data_dir == "data/multiwoz":
            fp_ontology = open(os.path.join(config.data_dir, "ontology.json"), "r")
            ontology = json.load(fp_ontology)
            for slot in ontology.keys():
                ontology[slot].append("none")
            fp_ontology.close()

            if not config.target_slot == 'all':
                slot_idx = {'attraction':'0:1:2', 'bus':'3:4:5:6', 'hospital':'7', 'hotel':'8:9:10:11:12:13:14:15:16:17',\
                            'restaurant':'18:19:20:21:22:23:24', 'taxi':'25:26:27:28', 'train':'29:30:31:32:33:34'}
                target_slot =[]
                prev_slot = []
                for key, value in slot_idx.items():
                    if key == config.target_slot:
                        target_slot.append(value)
                    else:
                        prev_slot.append(value)
                config.target_slot = ':'.join(target_slot)
                config.prev_slot = ':'.join(prev_slot)

        else:
            raise NotImplementedError()

        # sorting the ontology according to the alphabetic order of the slots
        self.ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        self.target_slot = []
        self.prev_slot = []

        self.target_slot_idx = sorted([ int(x) for x in config.target_slot.split(':')])
        self.prev_slot_idx = sorted([ int(x) for x in config.prev_slot.split(':')])

        ontology_items = list(self.ontology.items())
        for idx, domain in enumerate(ontology_items):
            slot, value = domain
            if slot == "pricerange":
                slot = "price range"

            if idx in self.target_slot_idx:
                self.target_slot.append(slot)
            elif idx in self.prev_slot_idx:
                self.prev_slot.append(slot)

        self.all_slot = self.prev_slot + self.target_slot

        logger.info('Processor: previous slots: ' + ', '.join(self.prev_slot))
        logger.info('Processor: target slots: '+ ', '.join(self.target_slot))

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
        return [ self.ontology[slot] for slot in self.target_slot]

    def get_prev_labels(self):
        """See base class."""
        return [ self.ontology[slot] for slot in self.prev_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
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

            label = [ line[4+idx] for idx in self.target_slot_idx]
            prev_label = [ line[4+idx] for idx in self.prev_slot_idx]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, prev_label=prev_label))
        return examples


def convert_examples_to_features(examples, label_list, prev_label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    slot_dim = len(label_list)
    prev_slot_dim = len(prev_label_list)

    def _hard_coding_label(label):
        return 'do not care' if label=='dontcare' else label

    def _get_label(label, label_list):
        label_id = []
        label_info = ''
        label_map = [{_label: i for i, _label in enumerate(labels)} for labels in label_list]

        for i, label in enumerate(label):
            label = _hard_coding_label(label)
            label_id.append(label_map[i][label])
            label_info += '%s (id = %d) ' % (label, label_map[i][label])
        return label_id, label_info

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
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        input_ids += [0] * (max_seq_length - len(input_ids)) # Note: padding idx = 0
        assert len(input_ids) == max_seq_length

        label_id, label_info = _get_label(example.label, label_list)
        prev_label_id, prev_label_info = _get_label(example.prev_label, prev_label_list)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
            logger.info("label: " + label_info)
            logger.info("previous label: " + prev_label_info)

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if (prev_dialogue_idx is not None) and (prev_dialogue_idx != curr_dialogue_idx):
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=[-1]*slot_dim,
                                           prev_label_id=[-1] * prev_slot_dim)] * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id,
                              prev_label_id=prev_label_id,
                              ))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=[-1]*slot_dim,
                                   prev_label_id=[-1]*prev_slot_dim)] * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_prev_label_ids = torch.tensor([f.prev_label_id for f in features], dtype=torch.long)

    # reshape tensors to [batch, turn, word]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)
    all_prev_label_ids = all_prev_label_ids.view(-1, max_turn_length, prev_slot_dim)

    return all_input_ids, all_input_len, all_label_ids, all_prev_label_ids


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
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


###############################################################################
# Miscellaneous functions
###############################################################################

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/gfs/nlp/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train: bert, bert-gru, bert-lstm, "
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--load_path', type=str, default='',
                        help='pretrained model directory name')
    parser.add_argument("--target_slot", default='', type=str, required=True,
                        help="Target slot idx to train model. ex. '0:1:2 or an excluding slot name 'attraction'" )
    parser.add_argument("--prev_slot", default='', type=str, required=True,
                        help="Previous trained slots. ex. '0:1:2 or an excluding slot name 'attraction'" )
    parser.add_argument("--tf_dir", default='tensorboard', type=str, required=False,
                        help="Tensorboard directory")
    parser.add_argument("--nbt", default='rnn', type=str, required=True,
                        help="nbt type: rnn or transformer or turn" )
    parser.add_argument("--fix_utterance_encoder",
                        action='store_true',
                        help="Do not train BERT utterance encoder")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_label_length", default=32, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length", default=22, type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--skip_connect',
                        type=str,
                        default=False,
                        help="skip-connection")
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
    parser.add_argument("--do_analyze",
                        action='store_true',
                        help="Whether to run analysis on the test set.")
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
    parser.add_argument("--lambda_ewc",
                        default=0.1,
                        type=float,
                        help="Hyper-parameter for EWC")
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
    tb_file_name = args.output_dir.split('/')[1]

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # CUDA setting
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

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(args)
    prev_label_list = processor.get_prev_labels()    # Slot value labels of Previous task
    target_label_list = processor.get_labels()       # Slot value labels of Present task
    label_list = prev_label_list + target_label_list # All slot value labels
    num_labels = [len(labels) for labels in label_list] # Number of labels of all slots
    #prev_slot_id = processor.prev_slot_idx
    #target_slot_id = processor.target_slot_idx
    # wrong
    prev_slot_id = list(range(0, len(processor.prev_slot)))  # List of slots in previous task
    target_slot_id = list(range(len(processor.prev_slot), len(processor.all_slot))) # list of slots in present task

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)
        num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
        num_dev_steps = int(len(dev_examples) / args.dev_batch_size * args.num_train_epochs)

        ## utterances
        all_input_ids, all_input_len, all_label_ids, all_prev_label_ids  = convert_examples_to_features(
            train_examples, target_label_list, prev_label_list, args.max_seq_length, tokenizer, args.max_turn_length)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len, all_label_ids, all_prev_label_ids \
            = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device), all_prev_label_ids.to(device)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids, all_prev_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Dev
        ## utterances
        all_input_ids_dev, all_input_len_dev, all_label_ids_dev, all_prev_label_ids_dev = convert_examples_to_features(
            dev_examples, target_label_list, prev_label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev, all_prev_label_ids_dev  = \
            all_input_ids_dev.to(device), all_input_len_dev.to(device), all_label_ids_dev.to(device), all_prev_label_ids_dev.to(device)

        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev, all_prev_label_ids_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt =='rnn':
        from BeliefTrackerSlotQueryMultiSlot import BeliefTracker
        if args.task_name.find("gru") == -1 and args.task_name.find("lstm") == -1:
            raise ValueError("Task name should include at least \"gru\" or \"lstm\"")

    elif args.nbt =='turn':
        from BeliefTrackerSlotQueryMultiSlotTurn import BeliefTracker

    elif args.nbt == 'transformer':
        from BeliefTrackerSlotQueryMultiSlotTransformer import BeliefTracker
        from BeliefTrackerSlotQueryMultiSlotEWC import EWC

    else:
        raise ValueError('nbt type should be either rnn or transformer')


    from BeliefTrackerSlotQueryMultiSlotEWC import EWC

    model = BeliefTracker(args, num_labels, device)
    if args.fp16:
        model.half()

    # Load pretrained model
    # in the case that slot and values are different between the training and evaluation
    ptr_model = torch.load(args.load_path, map_location=device)

    del_list = []
    rename_list = []
    for key in ptr_model.keys():
        if ('slot_lookup' in key) or ('value_lookup' in key): # remove slot_lookup and value_lookup
            del_list.append(key)
        if ('rnn.' in key): # rename rnn -> nbt,
            rename_list.append(key)
    for key in del_list:
        del ptr_model[key]
    for key in rename_list:
        new_key = key.replace('rnn.', 'nbt.')
        ptr_model[new_key] = ptr_model[key]
        del ptr_model[key]

    state = model.state_dict()
    state.update(ptr_model)
    model.load_state_dict(state)
    model.to(device)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get slot-type embeddings
    ## Note: slot embeddings are ordered as [previous slots + present target slots]
    slot_token_ids, slot_len = \
        get_label_embedding(processor.all_slot, args.max_label_length, tokenizer, device)

    model.initialize_slot_value_lookup(label_token_ids, label_len, slot_token_ids, slot_len)

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
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters

        if n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module)

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

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None

        #### EWC: calculate Fisher
        ewc = EWC(model, dev_dataloader, oldtask=prev_slot_id, num_labels=num_labels, device=device, n_gpu=n_gpu)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#        for epoch in trange(1):

            #### TRAIN
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids, _ = batch

                if n_gpu == 1:
                    loss_, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu, target_slot=target_slot_id)
                    loss_ewc = ewc.penalty(model)
                    loss = loss_ + args.lambda_ewc * loss_ewc
                else:
                    loss_, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu,
                                                               target_slot=target_slot_id)
                    loss_ = loss_.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                    loss_ewc = ewc.penalty(model)
                    loss = loss_ + args.lambda_ewc * loss_ewc

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss_, global_step)
                    summary_writer.add_scalar("Train/Loss_EWC", loss_ewc, global_step)
                    summary_writer.add_scalar("Train/Loss_Total", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    if n_gpu == 1:
                        for i, slot in enumerate(processor.target_slot):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ','_'), loss_slot[i], global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ','_'), acc_slot[i], global_step)

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
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            prev_dev_loss = 0
            prev_dev_acc = 0
            prev_dev_loss_slot, prev_dev_acc_slot = None, None
            prev_nb_dev_examples = 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids, prev_label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)
                    prev_label_ids = prev_label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if n_gpu == 1:
                        loss_, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu,
                                                                   target_slot=target_slot_id)
                        loss = loss_ + args.lambda_ewc * ewc.penalty(model)
                        prev_loss, prev_loss_slot, prev_acc, prev_acc_slot, _ = model(input_ids, input_len,
                                                                                      prev_label_ids, n_gpu,
                                                                                      target_slot=prev_slot_id)

                    else:
                        loss_, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu, target_slot=target_slot_id)
                        loss_ = loss_.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                        loss_ewc = ewc.penalty(model)
                        loss = loss_ + args.lambda_ewc * loss_ewc

                        prev_loss, _, prev_acc, prev_acc_slot, _ = model(input_ids, input_len, prev_label_ids, n_gpu, target_slot=prev_slot_id)
                        prev_loss = prev_loss.mean()
                        prev_acc = prev_acc.mean()
                        prev_acc_slot = prev_acc_slot.mean(0)

                num_valid_turn = torch.sum(label_ids[:,:,0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                prev_num_valid_turn = torch.sum(prev_label_ids[:,:,0].view(-1) > -1, 0).item()
                prev_dev_loss += prev_loss.item() * prev_num_valid_turn
                prev_dev_acc += prev_acc.item() * prev_num_valid_turn

                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [ l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                        prev_dev_loss_slot = [ l * prev_num_valid_turn for l in prev_loss_slot]
                        prev_dev_acc_slot = prev_acc_slot * prev_num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn
                        for i, l in enumerate(prev_loss_slot):
                            prev_dev_loss_slot[i] = prev_dev_loss_slot[i] + l * prev_num_valid_turn
                        prev_dev_acc_slot += prev_acc_slot * prev_num_valid_turn

                nb_dev_examples += num_valid_turn
                prev_nb_dev_examples += prev_num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples
            prev_dev_loss = prev_dev_loss / prev_nb_dev_examples
            prev_dev_acc = prev_dev_acc / prev_nb_dev_examples
            if n_gpu == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples
                prev_dev_acc_slot = prev_dev_acc_slot / prev_nb_dev_examples

            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                summary_writer.add_scalar("Validate/Prev_Loss", prev_dev_loss, global_step)
                summary_writer.add_scalar("Validate/Prev_Acc", prev_dev_acc, global_step)
                if n_gpu == 1:
                    for i, slot in enumerate(processor.target_slot):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ','_'), dev_loss_slot[i]/nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ','_'), dev_acc_slot[i], global_step)
                    for i, slot in enumerate(processor.prev_slot):
                        summary_writer.add_scalar("Validate/Prev_Loss_%s" % slot.replace(' ','_'), prev_dev_loss_slot[i]/prev_nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Prev_Acc_%s" % slot.replace(' ','_'), prev_dev_acc_slot[i], global_step)

            logger.info("*** Model Updated: Epoch=%d, Valid loss=%.6f, Valid acc=%.6f, Valid prev loss=%.6f, Valid prev acc=%.6f ***" \
                        % (epoch, dev_loss, dev_acc, prev_dev_loss, prev_dev_acc))

            dev_loss = round(dev_loss, 6)
            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (last_update, best_loss, best_acc))
            else:
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (epoch, dev_loss, dev_acc))

            #if epoch > 100 and last_update + args.patience <= epoch:
            if last_update + args.patience <= epoch:
                break

    ###############################################################################
    # Evaluation
    ###############################################################################

    # Test
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    # Load a trained model that you have fine-tuned
    ptr_model = torch.load(output_model_file, map_location=device)

    del_list = []
    for key in ptr_model.keys():
        if ('slot' in key) or ('value' in key):
            del_list.append(key)
    for key in del_list:
        del ptr_model[key]

    if n_gpu > 1:
        model = model.module

    state = model.state_dict()
    state.update(ptr_model)
    model.load_state_dict(state)
    model.to(device)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get slot-type embeddings
    ## Note: slot embeddings are ordered as [previous slots + present target slots]
    slot_token_ids, slot_len = \
        get_label_embedding(processor.all_slot, args.max_label_length, tokenizer, device)

    model.initialize_slot_value_lookup(label_token_ids, label_len, slot_token_ids, slot_len)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        all_input_ids, all_input_len, all_label_ids, all_prev_label_ids  = convert_examples_to_features(
            eval_examples, target_label_list, prev_label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_input_ids, all_input_len, all_label_ids, all_prev_label_ids \
            = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device), all_prev_label_ids.to(device)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids, all_prev_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        prev_eval_loss, prev_eval_accuracy = 0, 0
        prev_eval_loss_slot, prev_eval_acc_slot = None, None
        nb_eval_examples_prev = 0

        for input_ids, input_len, label_ids, prev_label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                label_ids = label_ids.unsuqeeze(0)
                prev_label_ids = prev_label_ids.unsuqeeze(0)

            with torch.no_grad():
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu, target_slot=target_slot_id)
                    prev_loss, prev_loss_slot, prev_acc, prev_acc_slot, _ = model(input_ids, input_len, prev_label_ids, n_gpu, target_slot=prev_slot_id)
                else:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu, target_slot=target_slot_id)
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                    prev_loss, prev_loss_slot, prev_acc, prev_acc_slot, _ = model(input_ids, input_len, prev_label_ids, n_gpu, target_slot=prev_slot_id)
                    prev_loss = prev_loss.mean()
                    prev_acc = prev_acc.mean()
                    prev_acc_slot = prev_acc_slot.mean(0)

            nb_eval_ex_prev = (prev_label_ids[:,:,0].view(-1) != -1).sum().item()
            nb_eval_examples_prev += nb_eval_ex_prev

            nb_eval_ex = (label_ids[:,:,0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1

            def _post_process(eval_loss, eval_loss_slot, eval_accuracy, eval_acc_slot, loss, loss_slot, acc, acc_slot, nb_eval_ex):
                eval_loss += loss.item() * nb_eval_ex
                eval_accuracy += acc.item() * nb_eval_ex
                if loss_slot is not None:
                    if eval_loss_slot is None:
                        eval_loss_slot = [ l * nb_eval_ex for l in loss_slot]
                    else:
                        for i, l in enumerate(loss_slot):
                            eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                if eval_acc_slot is None:
                    eval_acc_slot = acc_slot * nb_eval_ex
                else:
                    eval_acc_slot += acc_slot * nb_eval_ex

                return eval_loss, eval_loss_slot, eval_accuracy, eval_acc_slot

            eval_loss, eval_loss_slot, eval_accuracy, eval_acc_slot = \
                _post_process(eval_loss, eval_loss_slot, eval_accuracy, eval_acc_slot, loss, loss_slot, acc, acc_slot, nb_eval_ex)
            prev_eval_loss, prev_eval_loss_slot, prev_eval_accuracy, prev_eval_acc_slot = \
                _post_process(prev_eval_loss, prev_eval_loss_slot, prev_eval_accuracy, prev_eval_acc_slot, \
                              prev_loss, prev_loss_slot, prev_acc, prev_acc_slot, nb_eval_ex_prev)

        eval_loss /= nb_eval_examples
        if eval_loss_slot is None: # for multi-gpu
            eval_loss_slot = [0]
            prev_eval_loss_slot = [0]

        eval_accuracy = eval_accuracy / nb_eval_examples
        prev_eval_loss = prev_eval_loss / nb_eval_examples_prev

        prev_eval_accuracy = prev_eval_accuracy / nb_eval_examples_prev

        eval_acc_slot = eval_acc_slot / nb_eval_examples
        prev_eval_acc_slot = prev_eval_acc_slot / nb_eval_examples_prev

        total_acc_slot = {}

        for val, idx in zip(torch.cat([eval_acc_slot, prev_eval_acc_slot]), (target_slot_id+prev_slot_id)):
            total_acc_slot[idx] = val

        total_acc_slot = sorted(total_acc_slot.items(), key=operator.itemgetter(0))
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss,
                  'eval_loss_slot':'\t'.join([ str(val/ nb_eval_examples) for val in eval_loss_slot]),
                  'eval_acc_slot':'\t'.join([ str((val).item()) for val in eval_acc_slot]),
                  'prev_eval_loss': prev_eval_loss,
                  'prev_eval_accuracy': prev_eval_accuracy,
                  'prev_eval_loss_slot': '\t'.join([str(val / nb_eval_examples_prev) for val in prev_eval_loss_slot]),
                  'prev_eval_acc_slot': '\t'.join([str((val).item()) for val in prev_eval_acc_slot]),
                  'total_acc_slot': '\t'.join([str(val[1].item()) for val in total_acc_slot])
                  }

        out_file_name = 'eval_results'
        if args.target_slot=='all':
            out_file_name += '_all'
        output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    ###############################################################################
    # Analyze: TODO
    ###############################################################################
    if args.do_analyze and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        pdb.set_trace()
        def draw(data, x, y, ax):
            seaborn.heatmap(data,
                            xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                            cbar=False, ax=ax)

        class_correct = [[0 for x in range(num_labels[i])] for i in range(len(num_labels))]
        class_count = [[0 for x in range(num_labels[i])] for i in range(len(num_labels))]

        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(
            device), all_label_ids.to(device)
        logger.info("***** Running analysis *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", 1)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        model.eval()

        none_value_id = [ len(val)-1 for val in label_list]

        incorrect_dialogs = []
        attention_draw = 5
        for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                label_ids = label_ids.unsuqeeze(0)

            with torch.no_grad():
                _, _, acc, _, pred_slot = model(input_ids, input_len, label_ids, 1)


            nturn = (label_ids[:,:,0].view(-1) != -1).sum().item()
            nslot = label_ids.size(2)
            for slot in range(nslot):
                for turn in range(nturn):
                    class_count[slot][label_ids[0][turn][slot]]+=1
                    if label_ids[0][turn][slot] == pred_slot[0][turn][slot]:
                        class_correct[slot][label_ids[0][turn][slot]] +=1

            drawfig = False

            print('hotel')
            print(label_ids[0, 0:10, 8:18].cpu() == torch.Tensor(none_value_id[8:18]).long().repeat(10, 1))
            print(pred_slot[0, 0:10, 8:18].cpu() == torch.Tensor(none_value_id[8:18]).long().repeat(10, 1))
            print(label_ids[0, 0:10, 0:8].cpu() == torch.Tensor(none_value_id[0:8]).long().repeat(10, 1))
            print(label_ids[0, 0:10, 18:].cpu() == torch.Tensor(none_value_id[18:]).long().repeat(10, 1))

            pdb.set_trace()

            if drawfig == True:
            #if (len(incorrect_dialogs) < attention_draw):
                max_len = input_ids.size(2)
                attn_scores = model.attn.get_scores().transpose(1, 2).contiguous().view(label_ids.size(1)*nslot, -1, max_len)

                for slot in range(0, nslot):
                    fig, axs = plt.subplots(nturn, 1, figsize=(50, 10*nturn))
                    print("Slot", slot)
                    for turn in range(nturn):
                        draw(attn_scores[slot*label_ids.size(1)+turn,:,:].cpu(),
                             tokenizer.convert_ids_to_tokens(input_ids[0][turn].cpu().numpy()),
                             [*range(0, args.attn_head)], ax=axs[turn])
                        axs[turn].set_title("turn %d slot: %s label: %s pred: %s"
                                            % (turn, processor.target_slot[slot], str(label_list[slot][label_ids[0][turn][slot].item()]),
                                               str(label_list[slot][pred_slot[0][turn][slot].item()]) ))
                    plt.show()
                    plt.savefig(os.path.join(args.output_dir, "attention-d%d-slot%s.png"%(len(incorrect_dialogs), slot)))
                    plt.close()

            if not acc == 1:
                dialog = []
                for input, label, pred in zip(input_ids[0], label_ids[0], pred_slot[0]):
                    if label[0] == -1:
                        break
                    text = {}
                    text['input'] = ' '.join(tokenizer.convert_ids_to_tokens(input.cpu().numpy())).replace(' [PAD]', '')
                    text['label'] = [str(label_list[idx][x]) for idx, x in enumerate(label.cpu().numpy())]
                    text['pred'] = [str(label_list[idx][x]) for idx, x in enumerate(pred.cpu().numpy())]
                    dialog.append(text)

                incorrect_dialogs.append(dialog)

        output_eval_incorr_file = os.path.join(args.output_dir, "incorrect_dialog.txt")
        with open(output_eval_incorr_file, "w") as writer:
            for dialog in incorrect_dialogs:
                for turn in dialog:
                    text = turn['input'] + '\t'
                    for label, pred in zip(turn['label'], turn['pred']):
                        text += '%s\t%s\t'%(label, pred)
                    writer.write("%s\n" % text)
                writer.write("---------- \n")
        logger.info("Done analysis: %s" % output_eval_incorr_file)


        output_eval_incorr_file = os.path.join(args.output_dir, "per_class_accuracy.txt")
        with open(output_eval_incorr_file, "w") as writer:
            total_class_acc = 0
            total_slot_class_acc = []
            nlabels = 0
            for sid, slot in enumerate(class_count):
                slot_class_acc = 0
                for vid, value in enumerate(slot):
                    if not value == 0:
                        class_acc = class_correct[sid][vid]/value
                        writer.write("%s\t%d\t%d\t%.3f\n"%(label_list[sid][vid], class_correct[sid][vid], value, class_acc) )
                        slot_class_acc += class_acc
                        nlabels += 1
                    else:
                        writer.write("%s\t%d\t%d\t%.3f\n"%(label_list[sid][vid], class_correct[sid][vid], value, -1) )

                total_slot_class_acc.append(slot_class_acc/(vid+1))
                total_class_acc+=slot_class_acc
            total_class_acc /= nlabels

            for sid, slot_acc in enumerate(total_slot_class_acc):
                writer.write("%d\t%.3f\n" % (sid, slot_acc))
            writer.write("total class accuracy \t%.3f\n" % total_class_acc)

        logger.info("Done analysis: %s" % output_eval_incorr_file)

        print(class_correct)
        print(class_count)

if __name__ == "__main__":
    main()