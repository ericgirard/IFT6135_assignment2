import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class RNNStackUnit(nn.Module):
    """
    A helper class that encapsulates a single hidden unit in the RNN stack.
    The main class calls this class hidden_size times and calls its forward method
    one after the other
    """
    def __init__(self, rnn_input_size, hidden_size, batch_size, dp_keep_prob ):
        """
        rnn_input_size: The input dimensions of the unit. Differs only for the first
                        one which has its input the embedding.
        hidden_size: Hidden size as passed to the main RNN class
        batch_size: How many sequences per batch
        dp_keep_prob: Dropout keep probability
        """
        super(RNNStackUnit, self).__init__()
        #Parameters initialization.
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dp_keep_prob = dp_keep_prob
        self.output = None

        self.W_hx    = nn.Parameter(torch.Tensor(rnn_input_size, hidden_size))
        self.W_hh    = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h     = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(1-dp_keep_prob)
        self.state   = torch.Tensor(self.batch_size, hidden_size)
        self.state_tanh = nn.Tanh()

    def load_state(self, state):
        """
        A helper function to load the passed state into the stack. Mainly used
        at the initialization passed from the main RNN class.
        """
        self.state = state

    def init_weights(self):
        """
        Weight initialization as specified.
        """
        k = np.sqrt(1/self.hidden_size)
        torch.nn.init.uniform_( self.W_hx, a=-k, b=k)
        torch.nn.init.uniform_( self.W_hh, a=-k, b=k)
        torch.nn.init.uniform_( self.b_h, a=-k, b=k)

    def forward(self, X):
        """
        Computation of the RNN module.
        X: Current input sequence in batches.
        """
        self.state = self.state_tanh( (X@self.W_hx ) + (self.state@self.W_hh) + self.b_h )
        self.output = self.dropout(self.state)
        return self.output

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()
        #Save init parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        #define layers
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.embedding_dropout = nn.Dropout(1-dp_keep_prob)

        self.out = nn.Linear(self.hidden_size, vocab_size)

        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        self.rnns_stack = nn.ModuleList()

        #Create num_layers modules. For the first one, set the input size to be
        #the embedding size, otherwise their input is the hidden_size coming from
        #the previous unit in the stack.
        for l in range(self.num_layers):
            if l == 0:
                rnn_input_size = emb_size
            else:
                rnn_input_size = hidden_size

            self.rnns_stack.append(RNNStackUnit(rnn_input_size, hidden_size, batch_size, dp_keep_prob))

        self.init_weights()

    def init_weights(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        torch.nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.out.weight, a=-0.1, b=0.1)

        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        torch.nn.init.constant_(self.out.bias, val=0.0)

        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size
        k = np.sqrt(1/self.hidden_size)
        for l in range(self.num_layers):
            self.rnns_stack[l].init_weights()

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        #return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)) for i in self.num_layers)
        return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, inputs, hidden):
        """
        This function initalizes the hidden states of the stack modules,
        iterates over the sequence, and for each token cascades the
        flow of computation through the stack modules.
        """
        #Initialize hidden states
        for s in range(self.num_layers):=
            self.rnns_stack[s].load_state(hidden[s,:,:])

        #Loop over sequence tokens.
        for token_index in range(self.seq_len):
            current_input = self.embedding(inputs[token_index,:])
            current_input = self.embedding_dropout(current_input)
            #Loop over stack
            for s in range(self.num_layers):
                current_input = self.rnns_stack[s](current_input)

            #Compute output: **Do NOT apply softmax to the outputs!**
            current_input = self.out(current_input)

            #Construct the output logits as stacked outputs.
            if token_index == 0:
                logits = current_input
            else:
                logits = torch.cat([logits, current_input], 0)

        #Stack the current hidden states to return them
        hidden = torch.stack([self.rnns_stack[s].state for s in range(self.num_layers)])
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        #Samples is created as a list, so that we can have each token generated
        #as one element of the list (Note each element is a batch of tokens since
        #generation is done in batches.)
        samples=[inputs]

        #Load state
        for s in range(self.num_layers):
            self.rnns_stack[s].load_state(hidden[s,:,:])

        #No gradient calculations, save memory
        with torch.no_grad():
            #Loop for the passed sequence length
            for i in range(generated_seq_len):
                #Generate the next token.
                current_input = self.embedding(inputs)
                for s in range(self.num_layers):
                    current_input = self.rnns_stack[s](current_input)

                current_input = self.out(current_input)
                next_word_probs = F.softmax(current_input, -1)
                #Adding this line improves the output, suggestec by Tegan for controlling the "heat"
                next_word_probs = next_word_probs.exp()

                #Sample one of the words, not necessarily the one with highest prob
                chosen_words = torch.multinomial(next_word_probs, 1)

                #Append the chosen word and continue
                samples.append(chosen_words)
                current_input = chosen_words
        return samples



class GRUStackUnit(nn.Module):
    """
    A helper class that encapsulates a single hidden unit in the GRU stack.
    The main class calls this class hidden_size times and calls its forward method
    one after the other
    """
    def __init__(self, rnn_input_size, hidden_size, batch_size, dp_keep_prob ):
        super(GRUStackUnit, self).__init__()
        #Internal parameters initialization. These translate the inputs of the main
        #GRU class.
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dp_keep_prob = dp_keep_prob
        self.output = None

        self.W_rx    = nn.Parameter(torch.Tensor(rnn_input_size, hidden_size))
        self.W_rh    = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r     = nn.Parameter(torch.Tensor(hidden_size))

        self.W_zx    = nn.Parameter(torch.Tensor(rnn_input_size, hidden_size))
        self.W_zh    = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z     = nn.Parameter(torch.Tensor(hidden_size))

        self.W_htx    = nn.Parameter(torch.Tensor(rnn_input_size, hidden_size))
        self.W_hth    = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ht     = nn.Parameter(torch.Tensor(hidden_size))

        self.dropout = nn.Dropout(1-dp_keep_prob)

        self.rt   = torch.Tensor(self.rnn_input_size, hidden_size)
        self.zt   = torch.Tensor(self.rnn_input_size, hidden_size)
        self.state_tilde   = torch.Tensor(self.batch_size, hidden_size)
        self.state   = torch.Tensor(self.batch_size, hidden_size)
        self.rt_sigmoid = nn.Sigmoid()
        self.zt_sigmoig = nn.Sigmoid()
        self.state_tilde_tanh = nn.Tanh()

    def load_state(self, state):
        """
        A helper function to load the passed internal state into the unit.
        """
        self.state = state

    def init_weights(self):
        """
        The initializaztion of weights and biases, as specified.
        """
        k = np.sqrt(1/self.hidden_size)
        torch.nn.init.uniform_( self.W_rx, a=-k, b=k)
        torch.nn.init.uniform_( self.W_rh, a=-k, b=k)
        torch.nn.init.uniform_( self.b_r, a=-k, b=k)

        torch.nn.init.uniform_( self.W_zx, a=-k, b=k)
        torch.nn.init.uniform_( self.W_zh, a=-k, b=k)
        torch.nn.init.uniform_( self.b_z, a=-k, b=k)

        torch.nn.init.uniform_( self.W_htx, a=-k, b=k)
        torch.nn.init.uniform_( self.W_hth, a=-k, b=k)
        torch.nn.init.uniform_( self.b_ht, a=-k, b=k)

    def forward(self, X):
        """
        The forward method of the unit. Meant to be called by the main class.
        """
        self.rt = self.rt_sigmoid( (X@self.W_rx ) + (self.state@self.W_rh) + self.b_r )
        self.zt = self.zt_sigmoig( (X@self.W_zx ) + (self.state@self.W_zh) + self.b_z )

        self.state_tilde = self.state_tilde_tanh( (X@self.W_htx ) + ((self.rt*self.state)@self.W_hth) + self.b_ht )
        self.state = ( (1 - self.zt)*self.state ) + (self.zt*self.state_tilde)
        self.output = self.dropout(self.state)

        return self.output


class GRU(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(GRU, self).__init__()
        #Save init parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.embedding_dropout = nn.Dropout(1-dp_keep_prob)

        self.out = nn.Linear(self.hidden_size, vocab_size)

        self.grus_stack = nn.ModuleList()

        #Create the stack units, and take care of the input dimensions since
        #they differ if the unit is the first unit of the stak
        for l in range(self.num_layers):
            if l == 0:
                rnn_input_size = emb_size
            else:
                rnn_input_size = hidden_size

            self.grus_stack.append(GRUStackUnit(rnn_input_size, hidden_size, batch_size, dp_keep_prob))

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights and biases as specified.
        """
        torch.nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.out.weight, a=-0.1, b=0.1)
        torch.nn.init.constant_(self.out.bias, val=0.0)

        for l in range(self.num_layers):
            self.grus_stack[l].init_weights()

    def init_hidden(self):
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, inputs, hidden):
        """
        The forward method. This method takes the input sequences in batches, and
        the desired hidden state to start with, and then loads these states into the
        stack and makes a forward loop on the input
        """
        #Initialize the stack hidden state.
        for s in range(self.num_layers):
            self.grus_stack[s].load_state(hidden[s,:,:])

        #Loop over the tokens and propagate forward the computations.
        for token_index in range(self.seq_len):
            current_input = self.embedding(inputs[token_index,:])
            current_input = self.embedding_dropout(current_input)
            #Propagate the inputs through the stack,
            for s in range(self.num_layers):
                current_input = self.grus_stack[s](current_input)

            current_input = self.out(current_input)

            #Stack the inputs of current token with previous ones.
            if token_index == 0:
                logits = current_input
            else:
                logits = torch.cat([logits, current_input], 0)

        #Stack the current hidden state
        hidden = torch.stack([self.grus_stack[s].state for s in range(self.num_layers)])

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        samples=[inputs]

        #Load state
        for s in range(self.num_layers):
            self.grus_stack[s].load_state(hidden[s,:,:])

        #No gradient calculations, save memory
        with torch.no_grad():
            #Loop for the passed sequence length
            for i in range(generated_seq_len):
                #Generate the next token.
                current_input = self.embedding(inputs)
                for s in range(self.num_layers):
                    current_input = self.grus_stack[s](current_input)

                current_input = self.out(current_input)
                next_word_probs = F.softmax(current_input, -1)
                #Adding this line improves the output
                next_word_probs = next_word_probs.exp()

                #Sample a next word according to their probabilities
                chosen_words = torch.multinomial(next_word_probs, 1)

                samples.append(chosen_words)
                current_input = chosen_words
        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads

        self.n_heads = n_heads
        ]self.dropout_prob = dropout

        assert n_units % n_heads == 0
        self.n_units = n_units
        self.dropout = nn.Dropout(self.dropout_prob)

        #Note: Our original implementation did not involve using the clones function,
        #we just declared each as a  nn.Linear(n_units, n_units). We noticed a difference
        #in behaviour, so we opted to the one that yielded closer results to what was
        #specified. However, we kept the format that we think yields a more readable
        #code, although it was not necessary to do that
        self.linears = clones(nn.Linear(n_units, n_units), 4)

        self.QW = self.linears[0]
        self.KW = self.linears[1]
        self.VW = self.linears[2]
        self.out = self.linears[3]


    def init_weights(self):
        """
        Initialize the weights as specified.
        """
        k = np.sqrt(1/self.n_units)

        torch.nn.init.uniform_(self.out.weight, a=-k, b=k)
        torch.nn.init.constant_(self.out.bias, val=0.0)

        #-------------------------------
        torch.nn.init.uniform_(self.linears[0].weight, a=-k, b=k)
        torch.nn.init.constant_(self.linears[0].bias, val=0.0)

        torch.nn.init.uniform_(self.linears[1].weight, a=-k, b=k)
        torch.nn.init.constant_(self.linears[1].bias, val=0.0)

        torch.nn.init.uniform_(self.linears[2].weight, a=-k, b=k)
        torch.nn.init.constant_(self.linears[2].bias, val=0.0)

    def forward(self, query, key, value, mask=None):
        """
        The forward method receives the Q, K, V and mask, and performs the computations
        of the self attention.
        """
        batch_size =  query.size(0)
        seq_len = query.size(1)
        assert self.n_units == query.size(2)

        #Taken from blog
        if mask is not None:
            mask = mask.unsqueeze(1)

        #Resizing taken from blog, i.e. making it into 4D tensor then taking it back into 3D
        #Our original implementation used 3D directly, otherwise the code structure is our
        #original one
        self.Q = self.linears[0](query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        self.K = self.linears[1](key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        self.V = self.linears[2](value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        A = (torch.matmul(self.Q, self.K.transpose(-2, -1) )/(self.sqrt_d_k))
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)

        H = F.softmax(A, dim = -1)

        H = self.dropout(H)
        attention = torch.matmul(H, self.V )
        #Coming again from the Harvard blog style of arranging dimensions.
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_units)
        out = self.out(attention)
        return out



####################################################################################
#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
