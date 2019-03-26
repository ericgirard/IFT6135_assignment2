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
import json

from  utils import ptb_raw_data, load_model, get_model_seqgen_seed, convert_to_words, id_seq_2_word, repackage_hidden, Batch, ptb_iterator

"""
Note: this code only works for the RNN module, and is not complete. We have only included
it to reference our attempt to solve the problem 5.2
"""
#########################
#params of the file.
#Where the data is located.
data_path = './data'
#Where the models are saved. Models are expected to be saved in a subfolder in this path,
#Holding the name of the model passed in the variable model_name
model_path = './models'
#Name of the folder containing the 'best_params.pt' file.
model_name = 'GRU'

seq_len = 35
seed = 1111

#Setup the environment
np.random.seed(seed)
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

#Load model
model = load_model(model_path, model_name, vocab_size)
model = model.to(device)

#Load Validation
val_iterator = ptb_iterator(valid_data, model.batch_size, model.seq_len)
t_losses = [0 for t in range(seq_len)]

loss_fn = nn.CrossEntropyLoss()

print("Model Batch Size:",  model.batch_size)
print("Model SeqLen:", model.seq_len)
#Compute losses
for step, (x, y) in enumerate(val_iterator):
    #To TorchTensor then load to device
    if model_name != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = repackage_hidden(hidden)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        model.zero_grad()
        outputs, hidden = model(inputs, hidden)
    else:
        batch = Batch(torch.from_numpy(x).long().to(device))
        model.zero_grad()
        outputs = model.forward(batch.data, batch.mask).transpose(1,0)

    targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)

    for t in range(model.seq_len):
        step_loss = loss_fn(outputs[t, :], targets[t, :])
        t_losses[t] += int(step_loss)

#Average the losses
for t in range(model.seq_len):
    t_losses[t] /= (step)

#save the result as a dict
avg_loss_dict = {i:t_losses[i] for i in range(len(t_losses))}

#Write result to file
json.dump(avg_loss_dict, open(model_name+'_avggrad.txt','w'))
