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

from  utils import ptb_raw_data, load_model, get_model_seqgen_seed, convert_to_words, id_seq_2_word


#########################
#params of the file.
#Where the data is located.
data_path = './data'
#Where the models are saved. Models are expected to be saved in a subfolder in this path,
#Holding the name of the model passed in the variable model_name
model_path = './models'
#Name of the folder containing the 'best_params.pt' file.
model_name = 'GRU'

seq_len = 70
seed = 98766

#Note: only set this one to compare the generated texts between models. Setting it
#generates the exact same text for different seq_len for example, which is boring.
np.random.seed(seed)

#Setup the environment
torch.manual_seed(seed)
# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


#Load Data
raw_data = ptb_raw_data(data_path=data_path)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

#Load model
model = load_model(model_path, model_name, vocab_size)
model.eval()

print("MODEL LOADED SUCCESSFULLY!\n")


seed_words_np = numpy.random.randint(low=0, high=10000,size=model.batch_size)

inputs = seed_words_np

seed_words = torch.from_numpy(inputs).long()
hidden = model.init_hidden()

#Move everything to CUDA
model = model.to(device)
hidden = hidden.to(device)
seed_words = seed_words.to(device)

#Call the generate function.
generated_seqs = model.generate(seed_words, hidden, seq_len)
print(len(generated_seqs), len(generated_seqs[0]) )

#Convert the generated ids back to natural language tokens.
sentences_ids = []
for i, token in enumerate(generated_seqs):
    if i == 0:
        batch_tokens = token.cpu().numpy().tolist()
    else:
        batch_tokens = [item[0] for item in token.cpu().numpy().tolist()]
    sentences_ids.append(batch_tokens)

#Untie the generated batches into independent sequences.
sentences = list(zip(*sentences_ids))

print("Generated Sequences:")
print("====================")
generated_seqs_list = []
for i in sentences:
    generated_seq = id_seq_2_word(i, id_2_word)
    generated_seqs_list.append( generated_seq)
    print(generated_seq)
    print("\n\n")

#save the result to file.
with open(model_name+'_generated_'+str(seq_len)+'_seqlen.txt', mode='wt', encoding='utf-8') as f:
    f.write('\n\n'.join(generated_seqs_list))
