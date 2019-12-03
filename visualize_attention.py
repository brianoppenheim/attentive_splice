#from google.colab import drive
#drive.mount('/content/drive')

import torch
#!pip3 install transformers
#import transformers

from transformers import BertTokenizer, BertModel, BertConfig, BertForTokenClassification

from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from keras.preprocessing.sequence import pad_sequences

def read_in_val_data(filepath):
  print("reading ",filepath)
  genes = []
  labels = []
  df = pd.read_csv(filepath,usecols=[2,3],sep="\t",skiprows=1,header=None,nrows=10)
  print(df.head())
  for entry in df.itertuples():
    kmer_list = [kmer.strip("\'") for kmer in entry[1][1:-1].split(", ")]
    #for some reason there are empty examples in here
    if(len(kmer_list)>1):
      label_list = list(map(float, entry[2][1:-1].split(", ")))
      genes.append(kmer_list)
      labels.append(label_list)
  return genes, labels

def tokenize_and_pad_single_sample(sample,label):
  k= len(genes[0][0])
  if k==4:
    kmer_filepath = './kmers/fourmers.txt'
  elif k==6:
    kmer_filepath = './kmers/hexamers.txt'
  elif k==8:
    kmer_filepath = './kmers/octamers.txt'

  formatted_sample = ['[CLS]']  + sample + ['[SEP]']
  formatted_label = [0] + label + [0]
  tokenizer = BertTokenizer(kmer_filepath, max_len=MAX_LEN)
  attention_mask = np.concatenate([np.ones(len(label)), np.zeros(MAX_LEN - len(label))])
  seq_ids =[tokenizer.convert_tokens_to_ids(formatted_sample)]
  sample_ids = pad_sequences(seq_ids, maxlen=MAX_LEN, truncating='post', padding='post')[0]
  
  return sample_ids, attention_mask, formatted_label

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    #ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def top_kmers(windows, genes, labels):
    kmer_dict = dict()
    for i in range(windows):
        input_ids, mask, _ = tokenize_and_pad_single_sample(genes[i],labels[i])
        input_tensor = torch.tensor(input_ids).unsqueeze(0).long()
        mask_tensor = torch.tensor(mask).unsqueeze(0).long()
        _, attention = model(input_tensor, attention_mask=mask_tensor)
        attention_weights = attention[0].squeeze(0).detach().numpy()[0]
        max_column = np.argsort(np.mean(attention_weights, axis=0))[-5:]
        for m in max_column:
            seq = genes[i][m]
            if seq not in kmer_dict:
                kmer_dict[seq] = 0
            else:
                kmer_dict[seq] = kmer_dict[seq] + 1
    for k, v in kmer_dict.items():
        print(k + ': ' + str(v))

#FOR MSE
class BertForTokenRegression(torch.nn.Module):
  def __init__(self):
    super(BertForTokenRegression, self).__init__()
    self.config = BertConfig(vocab_size_or_config_json_file='./regression_bert_configuration.json')
    self.bert = BertForTokenClassification(self.config)
    self.linear = torch.nn.Linear(self.config.hidden_size, 1)
  def forward(self, input_ids, attention_mask=None):
    scores, hidden_states, att_weights = self.bert(input_ids,attention_mask=attention_mask)
    preds = self.linear(hidden_states[8]).squeeze(0).squeeze(1)
    return preds, att_weights

kmer_filepath='/content/drive/My Drive/kmers/hexamers.txt'
MAX_LEN=1002
model = BertForTokenRegression()
model.load_state_dict(torch.load("./brain_bert_weights.pt",map_location=torch.device('cpu')))

genes, labels = read_in_val_data('./Brain_train.txt')

window_number = 2
input_ids, mask, label = tokenize_and_pad_single_sample(genes[window_number],labels[window_number])
kmer_range = 150
splice_sites = np.where(np.array(label) > 0.0)
selected_splice_site = splice_sites[0][0]
print(splice_sites)
print(selected_splice_site)
min_kmer = np.maximum(0, selected_splice_site - kmer_range)
max_kmer = np.minimum(1002, selected_splice_site + kmer_range)
print(min_kmer)
print(max_kmer)
input_tensor = torch.tensor(input_ids).unsqueeze(0).long()
mask_tensor = torch.tensor(mask).unsqueeze(0).long()
preds, attention = model(input_tensor, attention_mask=mask_tensor)

att_path = "./attention_weights.txt"
attention_weights = attention[0].squeeze(0).detach().numpy()[0][min_kmer:max_kmer, min_kmer:max_kmer]
print(attention_weights.shape)
max_column = np.argsort(np.mean(attention_weights, axis=0))[-5:]
print(max_column)

print(len(genes))
top_kmers(10, genes, labels)
sequence_values_x = genes[window_number][min_kmer:max_kmer]
for m in max_column:
    print(sequence_values_x[m])
sequence_values_y = genes[window_number][min_kmer:max_kmer]
splice_site_indicator = [' ' for i in range(kmer_range*2)]
splice_site_indicator[selected_splice_site] = 'Splice'

fig, ax = plt.subplots()
im = ax.imshow(attention_weights)
im, cbar = heatmap(attention_weights, splice_site_indicator, splice_site_indicator, ax=ax,
                   cmap="GnBu", cbarlabel="attention")
#annotate_heatmap(im, valfmt="{x:.1f} t")

#fig.tight_layout()
#plt.savefig('correlation_plot.png', format="png")
fig.set_size_inches(10.5, 10.5)
plt.show()
