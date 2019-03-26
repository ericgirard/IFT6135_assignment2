import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
from models import RNN, GRU
from models import make_model as TRANSFORMER

"""
These are utility functions used by the tgenerator.py and avg_hidden_gradient.py
files. They are mostly functions copied from the ptb-lm.py file, along with a function
to load a saved model and another to generate a seed from the beginning of the sequences.
This function's purpose was to create realistic beginnings of the seed, however,
we did not use that one, as it generated too many 'the' (we should have foreseen that),
so the generated texts were not diverse. We ended up at the end just generating
any random words as seeds.
"""


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

def load_model(model_path, model_name, vocab_size):
    if model_name == 'RNN':
        model_class = RNN
        init_params = ['emb_size', 'hidden_size', 'seq_len', 'batch_size', 'vocab_size', 'num_layers', 'dp_keep_prob']
    elif model_name == 'GRU':
        model_class = GRU
        init_params = ['emb_size', 'hidden_size', 'seq_len', 'batch_size', 'vocab_size', 'num_layers', 'dp_keep_prob']
    elif model_name == "TRANSFORMER":
        model_class = TRANSFORMER
        init_params = ['vocab_size', 'n_units', 'n_blocks', 'dropout']
    else:
        print("Model name is incorrect, please use either RNN, GRU or TRANSFORMER")

    config_file_path = os.path.join(model_path, model_name, "exp_config.txt")
    with open(config_file_path) as file: # Use file to refer to the file object
        params = file.read().splitlines()

    model_init_params_dict = {}

    for entry in params:
        for param in init_params:
            cleaned_entry = entry.split("    ")
            if entry.startswith(param):
                if cleaned_entry[0] == 'dp_keep_prob':
                    model_init_params_dict[param] = float(cleaned_entry[1])
                else:
                    model_init_params_dict[param] = int(cleaned_entry[1])

    checkpoint = torch.load(os.path.join(model_path, model_name, "best_params.pt"))
    model = model_class(**model_init_params_dict, vocab_size=vocab_size)
    #print(checkpoint.keys() )

    #https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

    d = dict(checkpoint.items() )

    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    if model_name == "TRANSFORMER":
        model.batch_size = 20
        model.seq_len = 35
        model.vocab_size=vocab_size

    return model

def get_model_seqgen_seed(data, model):
    x, y = next(ptb_iterator(data, model.batch_size, model.seq_len))
    #print(x)
    #print(x.shape)
    x = torch.from_numpy(x).long()
    mask = torch.zeros_like(x)
    #batch shape: (seq_len, batch_size)
    #print("Mask Shape", mask.size() )
    mask[0,1] = 0
    #mask x
    x = x.masked_fill(mask == 0, 0.0)
    return x, y

def convert_to_words(gen_tensor, id_2_word):
    sentences = []
    return sentences


def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

def id_seq_2_word(id_list, id_2_word):
    return " ".join([id_2_word[i] for i in id_list])



def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)
