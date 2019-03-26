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


#########################
#params
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
model.train()

#Initialize the lists where the losses per token will be saved.
#These lists will be used by the hooks
hidden_1_losses_list = []
hidden_2_losses_list = []


#Hooks
def hidden_1_losses(h):
    global hidden_1_losses_list
    l = h
    l2_norm = torch.norm(l, p=2, dim=1)
    hidden_1_losses_list.append(l2_norm)

def hidden_2_losses(h):
    global hidden_2_losses_list
    l = h
    l2_norm = torch.norm(l, p=2, dim=1)
    hidden_2_losses_list.append(l2_norm)

if model_name == "RNN":
    model.rnns_stack[0].W_hh.register_hook(hidden_1_losses)
    model.rnns_stack[1].W_hh.register_hook(hidden_2_losses)
elif model_name == "GRU":
    model.grus_stack[0].W_hh.register_hook(hidden_1_losses)
    model.grus_stack[1].W_hh.register_hook(hidden_2_losses)

#Load Validation
val_iterator = ptb_iterator(valid_data, model.batch_size, model.seq_len)
t_losses = [0 for t in range(seq_len)]

loss_fn = nn.CrossEntropyLoss()

print("Model Batch Size:",  model.batch_size)
print("Model SeqLen:", model.seq_len)

#Compute losses on 3 iterations
n_iterations = 3

for step, (x, y) in enumerate(val_iterator):
    #Take care if the input, taken from the code provided.
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

    print("Step: ",step)
    if step == n_iterations:
        break
    for t in range(model.seq_len
        step_loss = loss_fn(outputs[t, :], targets[t, :])
        step_loss.backward(retain_graph=True)


avg_l2_grad = [0 for i in range(seq_len)]

#So all the computations are saved into the list. We want to adjust so that
#each value at t time step is computed together
for i in range(seq_len*n_iterations):
    print(i, i%seq_len)
    avg_l2_grad[i%seq_len] += float(hidden_2_losses_list[i].mean() )

#Compute the average per number of iterations.
for i in range(seq_len):
    avg_l2_grad[i] = avg_l2_grad[i]/n_iterations

#Get average in terms of number steps
for t in range(model.seq_len):
    avg_l2_grad[t] /= (step)

#Get the result in a dictionary format, with key=seq t
avg_loss_dict = {i:avg_l2_grad[i] for i in range(len(avg_l2_grad))}


#Print the results
print(avg_l2_grad)
#Write result to file
json.dump(avg_loss_dict, open(model_name+'_avgloss.txt','w'))
