from collections import namedtuple
import json
from argparse import Namespace
from neural_srl.shared.measurements import Timer
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared import reader
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from neural_srl.theano.tagger import BiLSTMTaggerModel
from tqdm import tqdm
import sklearn.metrics
from datetime import datetime

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


Argv = namedtuple("Argv", "config dev gold labels model task train vocab")

args = Argv(config='../config/srl_small_config.json', dev='../data/srl/conll2012.devel.txt', gold='../data/srl/conll2012.devel.props.gold.txt', labels='', model='conll2012_small_model', task='srl', train='../data/srl/conll2012.train.txt', vocab='')

args.config

def get_config(config_filepath):
  with open(config_filepath, 'r') as config_file: 
    conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
  return conf


config = get_config(args.config)

i = 0
global_step = 0
epoch = 0
train_loss = 0.0

with Timer('Data loading'):
    vocab_path = args.vocab if args.vocab != '' else None
    label_path = args.labels if args.labels != '' else None
    gold_props_path = args.gold if args.gold != '' else None
    
    print ('Task: {}'.format(args.task))
    
    data = TaggerData(config,
                        *reader.get_srl_data(config, args.train, args.dev, vocab_path, label_path))
    
    batched_dev_data = data.get_development_data(batch_size=config.dev_batch_size)
    print ('Dev data has {} batches.'.format(len(batched_dev_data)))


with Timer('Preparation'):
    if not os.path.isdir(args.model):
      print ('Directory {} does not exist. Creating new.'.format(args.model))
      os.makedirs(args.model)
    else:
      if len(os.listdir(args.model)) > 0:
        print ('[WARNING] Log directory {} is not empty, previous checkpoints might be overwritten'
             .format(args.model))
    shutil.copyfile(args.config, os.path.join(args.model, 'config'))
    # Save word and label dict to model directory.
    data.word_dict.save(os.path.join(args.model, 'word_dict'))
    data.label_dict.save(os.path.join(args.model, 'label_dict'))
    writer = open(os.path.join(args.model, 'checkpoints.tsv'), 'w')
    writer.write('step\tdatetime\tdev_loss\tdev_accuracy\tbest_dev_accuracy\n')

class BIOModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layers, hidden_dim, pretrained_embedding):
        super(BIOModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.word_embed = nn.Embedding.from_pretrained(pretrained_embedding)
        self.mask_embed = nn.Embedding(2, embedding_dim, device=device)
        
        self.biLSTM = nn.LSTM(embedding_dim * 2, hidden_dim, n_layers, dropout=0.1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, data.label_dict.size())
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, sequence_length = x.size(0), x.size(1)
        x_w = self.word_embed(x[:,:,0])
        x_m = self.mask_embed(x[:,:,1])
        x = torch.concat((x_w, x_m), axis=2)
        
        hidden_state, cell_state = self.init_hidden(batch_size)
        
        output, (hidden_state, cell_state) = self.biLSTM(x, (hidden_state, cell_state))
        
        output = self.linear(output)
        scores = self.softmax(output)
        pred = torch.argmax(scores, dim=-1)
        
        return output, pred
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.n_layers * 2, batch_size, self.hidden_dim), device=device)
        c0 = torch.zeros((self.n_layers * 2, batch_size, self.hidden_dim), device=device)
        return h0, c0

pretrained_embedding = torch.tensor(data.embeddings[0], device=device)

bioModel = BIOModel(data.word_dict.size(), 
                    embedding_dim=len(data.embeddings[0][0]), 
                    n_layers=4, hidden_dim=config.lstm_hidden_size, 
                    pretrained_embedding=pretrained_embedding)

bioModel.to(device)

def loss(scores, y):
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(scores.permute((0,2,1)), y)

from torch.optim import Adadelta, Adam

optimizer = Adam(bioModel.parameters(), lr=0.01)

def evaluation(model):
    flatten_ys = []
    flatten_preds = []
    for i, batch in enumerate(tqdm(batched_dev_data)):
        x, y, sq_lengths, weights = batch
        x_tensor = torch.stack([torch.tensor(i, device=device) for i in x])
        y_tensor = torch.stack([torch.tensor(i, device=device) for i in y])

        scores, preds = bioModel(x_tensor)
        
        for i, y in enumerate(y_tensor):
            
            flatten_y_list = y_tensor[i][:sq_lengths[i]].cpu().tolist()
            flatten_pred = preds[i][:sq_lengths[i]].cpu().tolist()
            flatten_ys = flatten_ys + flatten_y_list
            flatten_preds = flatten_preds + flatten_pred

    f1 = sklearn.metrics.f1_score(flatten_ys, flatten_preds, average="macro")
    
    return f1

def train(model, optimizer, loss_fn):
    train_data = data.get_training_data(include_last_batch=True)
    best_f1 = 0
    for epoch in range(config.max_epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_data)):
            x, y, _, weights = batch
            x_tensor = torch.stack([torch.tensor(i, device=device) for i in x])
            y_tensor = torch.stack([torch.tensor(i, device=device) for i in y])
            
            model.train()
            optimizer.zero_grad()
            
            scores, preds = model(x_tensor)
            
            # print(scores.shape)
            # print(y_tensor.shape)
            # return
            
            loss = loss_fn(scores, y_tensor)
            train_loss += loss
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), float(config.max_grad_norm))
            optimizer.step()
            
        i += 1
        train_loss = train_loss / i
        print("Epoch {}, steps={}, loss={:.6f}".format(epoch + 1, i, train_loss))
        model.eval()
        f1_dev = evaluation(model)
        if f1_dev > best_f1:
            torch.save(model.state_dict(), os.path.join(args.model, f"out-epoch-{epoch}-{datetime.now().strftime('%d-%m-%Y_%H')}.pt"))
        print("Eval: macro f1={:.3f}, best f1={:.3f}".format(f1_dev, best_f1))

train(bioModel, optimizer, loss)