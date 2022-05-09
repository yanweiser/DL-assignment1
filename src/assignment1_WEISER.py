# -*- coding: utf-8 -*-
"""
   Deep Learning for NLP
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Remember to use PyTorch for your NN implementation.
   Original code by Hande Celikkanat & Miikka Silfverberg. Minor modifications by Sharid Loáiciga.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir


# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/) 
embeddings_file = 'GoogleNews-pruned2tweets.bin'


#--- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare! 
# You can turn in your assignment with the best settings you find.

n_classes = len(LABEL_INDICES)
n_epochs = 40
learning_rate = 3e-4
report_every = 1
verbose = False

my_device = torch.device('cpu')
# since the computation is not vectorized, its faster to run it on the cpu
# if torch.cuda.is_available():
#       my_device = torch.device('cuda')
#       cuda = True
# else:
#       my_device = torch.device('cpu')
#       cuda = False



#--- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
  return torch.LongTensor([LABEL_INDICES[label]]).to(my_device)

# convert Token to its embedding
def token_to_embed(token, tti, ite):
      # embed_np = ite[tti[token]]
      # print(embed_np)
      if token in tti:
        return torch.from_numpy(ite[tti.get(token)]).float().to(my_device)
      else:
        return torch.zeros((ite.shape[1])).float().to(my_device)



#--- model ---

class FFNN(nn.Module):
  # Feel free to add whichever arguments you like here.
  # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)
  def __init__(self, pretrained_embeds, n_classes, hid_size1=64, hid_size2=64):
      super(FFNN, self).__init__()
      self.l1 = nn.Linear(pretrained_embeds.shape[1], hid_size1, device=my_device)
      self.l2 = nn.Linear(hid_size1, hid_size2, device=my_device)
      self.l3 = nn.Linear(hid_size2, n_classes, device=my_device)
      self.relu_act = nn.ReLU()
      self.log_sm = nn.LogSoftmax(dim=0)
      

  def forward(self, x):
      h1 = self.relu_act(self.l1(x))
      h2 = self.relu_act(self.l2(h1))
      logits = self.log_sm(self.l3(h2))
      return logits


#--- "main" ---

if __name__=='__main__':
  #--- data loading ---
  data = read_semeval_datasets(data_dir)
  gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file), binary=True)
  pretrained_embeds = gensim_embeds.vectors
  # To convert words in the input tweet to indices of the embeddings matrix:
  word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}

  #--- set up ---
  # WRITE CODE HERE
  model = FFNN(pretrained_embeds, n_classes, hid_size1=128)
  loss_function = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  

  #--- training ---
  for epoch in range(n_epochs):
    training_losses = []
    total_loss = 0
    for tweet in data['training']: 
      gold_class = label_to_idx(tweet['SENTIMENT'])
      sent = torch.zeros((300), device=my_device).float()
      for token in tweet['BODY']:
            sent += token_to_embed(token, word_to_idx, pretrained_embeds)
      preds = model(sent)
      optimizer.zero_grad()
      loss = loss_function(torch.reshape(preds, (1,3)), gold_class)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    if ((epoch+1) % report_every) == 0:
      print('epoch: %d, loss: %.4f' % (epoch, total_loss*100/len(data['training'])))
    
  # Feel free to use the development data to tune hyperparameters if you like!

  #--- test ---
  correct = 0
  with torch.no_grad():
    for tweet in data['training']:
      gold_class = label_to_idx(tweet['SENTIMENT'])
    
      # WRITE CODE HERE
      sent = torch.zeros((300), device=my_device).float()
      for token in tweet['BODY']:
            sent += token_to_embed(token, word_to_idx, pretrained_embeds)
      preds = model(sent)
      predicted = torch.argmax(preds, dim=0)
      correct += torch.eq(predicted,gold_class).item()

      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], predicted))
        
    print('training accuracy: %.2f' % (100.0 * correct / len(data['training'])))


  correct = 0
  with torch.no_grad():
    for tweet in data['development.gold']:
      gold_class = label_to_idx(tweet['SENTIMENT'])
    
      # WRITE CODE HERE
      sent = torch.zeros((300), device=my_device).float()
      for token in tweet['BODY']:
            sent += token_to_embed(token, word_to_idx, pretrained_embeds)
      preds = model(sent)
      predicted = torch.argmax(preds, dim=0)
      correct += torch.eq(predicted,gold_class).item()

      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], predicted))
        
    print('development accuracy: %.2f' % (100.0 * correct / len(data['development.gold'])))


  correct = 0
  with torch.no_grad():
    for tweet in data['test.gold']:
      gold_class = label_to_idx(tweet['SENTIMENT'])
    
      # WRITE CODE HERE
      sent = torch.zeros((300), device=my_device).float()
      for token in tweet['BODY']:
            sent += token_to_embed(token, word_to_idx, pretrained_embeds)
      preds = model(sent)
      predicted = torch.argmax(preds, dim=0)
      correct += torch.eq(predicted,gold_class).item()

      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], predicted))
        
    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))


