
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

#TODO (need to figure out a solution for padding)
def read_data_from_split_file(filepath):
	print("reading ",filepath)
	genes = []
	labels = []
	df = pd.read_csv(filepath,usecols=[2,3],sep="\t",skiprows=1,header=None)
	print(df.head())
	for entry in df.itertuples():
		kmer_list = [kmer.strip("\'") for kmer in entry[1][1:-1].split(", ")]
		label_list = list(map(float, entry[2][1:-1].split(", ")))
		genes.append(kmer_list)
		labels.append(label_list)
	return genes, labels

def read_non_split_file(filepath):
	print("reading ",filepath)
	genes = []
	labels = []
	df = pd.read_csv(filepath,usecols=[1,2],sep="\t",skiprows=1,header=None)
	print(df.head())
	for entry in df.itertuples():
		kmer_list = [kmer.strip("\'") for kmer in entry[1][1:-1].split(", ")]
		label_list = list(map(float, entry[2][1:-1].split(", ")))
		genes.append(kmer_list)
		labels.append(label_list)
	return genes, labels



def tokenize_and_pad_samples(genes,labels):
  k= len(genes[0][0])
  if k==4:
    kmer_filepath = '/home/brian/Downloads/fourmers.txt'
  elif k==6:
    kmer_filepath = '/home/brian/Downloads/hexamers.txt'
  elif k==8:
    kmer_filepath = '/home/brian/Downloads/octamers.txt'
  formatted_samples = [['[CLS]']  + sample + ['[SEP]'] for sample in genes]
  formatted_labels = [[0] + l + [0] for l in labels]
  tokenizer = BertTokenizer(kmer_filepath, max_len=MAX_LEN)
  print("TOKENIZER LENGTH", len(tokenizer))
  attention_masks = [np.concatenate([np.ones(len(l)), np.zeros(MAX_LEN - len(l))]) for l in formatted_labels]
  #seq_ids = tokenizer.convert_tokens_to_ids(formatted_samples)
  seq_ids = [tokenizer.convert_tokens_to_ids(sample) for sample in formatted_samples]
  seq_ids = pad_sequences(seq_ids, maxlen=MAX_LEN, truncating='post', padding='post')
  
  return seq_ids, attention_masks, formatted_labels

def handle_tokenization(file_table):
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

MAX_LEN=16670
genes,labels = read_non_split_file('/home/brian/Downloads/all_samples_6-mer_train.txt')
seq_ids, masks, labels = tokenize_and_pad_samples(genes,labels)
print(seq_ids[0])
print(len(seq_ids))
print("Finished making data")

batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForTokenClassification(BertConfig.from_json_file('/home/brian/attentive_splice/bert_configuration_all_hex.json'))
model.resize_token_embeddings(4099)
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3) #lr=3e-5)
class_weights = torch.tensor(np.array([1.0, 165.0])).float().cuda()
loss = CrossEntropyLoss(weight=class_weights)
last_i = 0

def load_model_from_saved():
  with open('/home/brian/bert_last_i.txt', 'r') as last_i_file:
    i = last_i_file.read()
    last_i = int(i)
    model.load_state_dict(torch.load("/home/brian/bert_splice_weights.pt"))

def save_weights():
  print("Saving weights")
  path = "/home/brian/bert_weights_6mer.pt"
  last_seq_path = "/home/brian/bert_last_i.txt"
  with open(last_seq_path, 'w+') as seq_record:
    seq_record.write(str(batch))
  torch.save(model.state_dict(), path)

def run_epoch(input_ids,masks,labels):
  model.train()
  epoch_loss=0
  cycle_loss = 0
  num_samples = len(input_ids)
  for curr in range(last_i, num_samples// batch_size):
    curr_sample = torch.tensor(input_ids[curr]).unsqueeze(0).long().cuda()
    curr_masks = torch.tensor(masks[curr]).long().cuda()
    curr_labels = torch.tensor(labels[curr]).long().cuda()
    optimizer.zero_grad()
    predictions = model(curr_sample, attention_mask=curr_masks)
    l = loss(predictions[0].squeeze(0), curr_labels)
    cycle_loss += l.item()
    l.backward()
    optimizer.step()

    if batch > 0 and batch % 1000 == 0:
      save_weights()
    if batch > 0 and batch % 100 == 0:
      print("Batch: " + str(batch) + " loss: " + str(total_loss / 100))
      epoch_loss+=cycle_loss
      cycle_loss = 0
  epoch_loss+=cycle_loss
  print("Epoch Loss"+str(epoch_loss/(num_samples-last_i)))

def evaluate_model(input_ids,masks,labels):
  model.eval()
  num_samples=len(input_kmers)
  eval_loss=0
  for i in range(num_samples):
    seq_ids = input_ids[i]
    attention_masks = torch.tensor(masks[i]).long().cuda()
    labels = torch.tensor(labels[i]).long().cuda()
    with torch.no_grad():
      predictions = model(seq_ids, attention_mask=attention_masks)
      print(predictions.detach().cpu().numpy())
      l = loss(predictions[0].squeeze(0), labels[0])
      eval_loss += l.item()
  print("Evaluation Loss = "+str(eval_loss/num_samples))

run_epoch(seq_ids, masks, labels)

training_loss = []
model.train()
total_loss = 0
for batch in range(len(formatted_hexamers) // batch_size):
    seq_batch = formatted_hexamers[batch*batch_size:(batch+1)*batch_size]
    attention_masks_batch = attention_masks[batch*batch_size:(batch+1)*batch_size]
    attention_masks_batch = torch.tensor(attention_masks_batch).long().cuda()
    split_seq_batch = seq_batch[0].split()
    print(split_seq_batch)
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
    '''
    if batch > 0 and batch % 500 == 0:
      path = "/content/drive/My Drive/bert_splice_weights.pt"
      last_seq_path = "/content/drive/My Drive/bert_last_i.txt"
      with open(last_seq_path, 'w+') as seq_record:
        seq_record.write(str(batch))
      loss_trace_path = "/content/drive/My Drive/bert_loss_trace.txt"
      with open(loss_trace_path, 'a+') as loss_record:
        loss_record.write(str(batch) + ' loss: ' + str(total_loss/100) + '\n')
      torch.save(model.state_dict(), path)
      '''
    if batch > 0 and batch % 100 == 0:
      print("Batch: " + str(batch) + " loss: " + str(total_loss / 100))
      total_loss = 0
