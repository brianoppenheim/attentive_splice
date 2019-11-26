

import torch
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
from keras.preprocessing.sequence import pad_sequences
import sys
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, permutations
import datetime



n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
cuda0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(cuda0)


import psutil
import humanize
import os
import GPUtil as GPU

GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

def process_data(path):
    first = True
    file_table = []
    with open(path) as fp:
        i = 0
        for line in fp:
            if not first:
                new_entry = []
                split_line = line.split()
                new_entry.append(split_line[0])
                new_entry.append(split_line[1])
                sep = ' '
                hexamers = list(map(lambda x: x.replace(']', '').replace('[', '').replace(',','').replace('\'', ''), split_line[2:1002]))
                if '--PAD--' in hexamers:
                    hexamers = hexamers[:hexamers.index('--PAD--')]
                hexamers_string = sep.join(hexamers)
                new_entry.append(hexamers_string)
                new_entry.append(list(map(lambda x: int(x.replace(']', '').replace('[', '').replace(',','').replace('.0','').replace('\'','')), split_line[1002:])))
                file_table.append(new_entry)
                i += 1
            first = False
    
    return handle_tokenization(file_table)

def handle_tokenization(file_table):
    MAX_LEN = 1002
    fourmers_path = '/home/brian/Downloads/fourmers.txt'
    hexamers_path = '/home/brian/Downloads/hexamers.txt'
    octamers_path = '/home/brian/Downloads/octamers.txt'

    formatted_hexamers = ['[CLS] ' + f[2] + ' [SEP]' for f in file_table]
    labels = [[0] + f[3] + [0] for f in file_table]
    tokenizer = BertTokenizer(hexamers_path, max_len=MAX_LEN)
    #hexamer_set = generate_hexamer_tokens()
    #tokenizer.add_tokens(hexamer_set)
    attention_masks = [np.concatenate([np.ones((len(a.split()))), np.zeros((MAX_LEN - len(a.split())))]) for a in formatted_hexamers]
    
    return tokenizer, formatted_hexamers, attention_masks, labels


def generate_hexamer_tokens():
    nucleotides = ['A', 'T', 'G', 'C']
    hexamer_tuples = combinations_with_replacement(nucleotides, 6)
    hexamer_tokens = [''.join(hexamer_tuple) for hexamer_tuple in hexamer_tuples]
    hexamer_set = set()
    for token in hexamer_tokens:
        permuted_hexamers = [''.join(l) for l in list(permutations(token))]
        hexamer_set.update(permuted_hexamers)
    assert len(hexamer_set) == 4096
    return list(hexamer_set)

batch_size = 1
MAX_LEN = 1002

tokenizer, formatted_hexamers, attention_masks, labels = process_data('/home/brian/Downloads/split_samples_6-mer_train.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0) 

model = BertForTokenClassification(BertConfig.from_json_file('/home/brian/attentive_splice/bert_configuration_split_hex.json'))
model.resize_token_embeddings(len(tokenizer))
model.to(device)

#model.load_state_dict(torch.load("/content/drive/My Drive/bert_splice_weights.pt"))
last_i = 0
#with open('/home/brian/bert_last_i.txt', 'r') as last_i_file:
#  i = last_i_file.read()
#  last_i = int(i)

print(last_i)
optimizer = Adam(model.parameters(), lr=3e-5)

training_loss = []
model.train()
class_weights = torch.tensor(np.array([1.0, 165.0])).float().cuda()
loss = CrossEntropyLoss(weight=class_weights)
print("ready to train")
total_loss = 0
for batch in range(len(formatted_hexamers) // batch_size):
    seq_batch = formatted_hexamers[batch*batch_size:(batch+1)*batch_size]
    attention_masks_batch = attention_masks[batch*batch_size:(batch+1)*batch_size]
    attention_masks_batch = torch.tensor(attention_masks_batch).long().cuda()
    split_seq_batch = seq_batch[0].split()
    batch_ids = [tokenizer.convert_tokens_to_ids(split_seq_batch)]
    batch_ids = pad_sequences(batch_ids, maxlen=MAX_LEN, truncating='post', padding='post')[0]
    batch_ids = torch.tensor(batch_ids).unsqueeze(0).long().cuda()
    batch_labels = torch.tensor(labels[batch*batch_size:(batch+1)*batch_size]).long().cuda()

    optimizer.zero_grad()
    predictions = model(batch_ids, attention_mask=attention_masks_batch)
    l = loss(predictions[0].squeeze(0), batch_labels[0])
    total_loss += l.item()
    training_loss.append(l.item())
    l.backward()
    optimizer.step()
    if batch > 0 and batch % 500 == 0:
      path = "/home/brian/bert_splice_weights.pt"
      last_seq_path = "/home/brian/bert_last_i.txt"
      with open(last_seq_path, 'w+') as seq_record:
        seq_record.write(str(batch))
      loss_trace_path = "/home/brian/bert_loss_trace.txt"
      with open(loss_trace_path, 'a+') as loss_record:
        loss_record.write(str(batch) + ' loss: ' + str(total_loss/100) + '\n')
      torch.save(model.state_dict(), path)
    if batch > 0 and batch % 100 == 0:
      print("Batch: " + str(batch) + " loss: " + str(total_loss / 100))
      total_loss = 0


