

import torch

from transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLConfig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def read_data(filepath):
  print("reading ",filepath)
  genes = []
  labels = []
  df = pd.read_csv(filepath,usecols=[1,2],sep="\t",header=None,skiprows=1,nrows=100)
  print(df.head())
  for entry in df.itertuples():
    kmer_list = [kmer.strip("\'") for kmer in entry[1][1:-1].split(", ")]
    label_list = list(map(float, entry[2][1:-1].split(", ")))
    genes.append(kmer_list)
    labels.append(label_list)
  return genes, labels
	

def tokenize_samples(genes):
  k= len(genes[0][0])
  if k==2:
    kmer_filepath = '/Users/camillo_stuff/Downloads/fourmersXL.txt'
  elif k==6:
    kmer_filepath = '/Users/camillo_stuff/Downloads/hexamersXL.txt'

  tokenizer=TransfoXLTokenizer(vocab_file=kmer_filepath)
  print("TOKENIZER LENGTH", len(tokenizer))
  seq_ids = [tokenizer.convert_tokens_to_ids(gene) for gene in genes]
  return seq_ids

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

'''
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
'''

def get_attentions_for_sample(curr_kmers):
  mems=None
  atts_for_gene=[]
  #looping over all windows
  for w in range(0, len(curr_kmers), window_size):
    toks = curr_kmers[w:w+window_size]
    #ignore tiny windows(leads to dimensionality issues)
    if(len(toks)<2):
      continue
    window_input_ids = torch.tensor(toks).unsqueeze(0)
    with torch.no_grad():
      window_preds, mems,atts = model(window_input_ids, mems)
    atts_for_gene.append(atts)
  return np.array(atts_for_gene)

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.config = TransfoXLConfig(vocab_size_or_config_json_file='regression_XL_configuration.json')
    self.config.vocab_size=204098
    self.config.n_layers=3
    self.config.n_heads=5
    self.config.output_attentions=True
    self.model = TransfoXLModel(self.config)
    self.out_layer = torch.nn.Linear(self.model.d_model, 2)
  def forward(self, input_ids, mems=None):
    hidden_states, mems, atts = self.model(input_ids, mems)
    preds = self.out_layer(hidden_states[0]).squeeze(0)
    np_att = atts[0].squeeze(0).detach().numpy()
    print(np.shape(np_att[0]))
    return preds, mems, np_att[0]

#building model
model = Model()
f="/Users/camillo_stuff/Downloads/xl_classification_6merv11.pt"
model.load_state_dict(torch.load(f,map_location=torch.device('cpu')))
window_size=1012


genes,labels = read_data('/Users/camillo_stuff/Documents/AttentiveSplice/data/output/partitioned_samples/all_samples_6-mer/all_samples_6-mer_train.txt')
gene_ids = tokenize_samples(genes)
print(len(gene_ids))
print("Finished making data")

curr_sample=29
window_num = 0
first_kmer = 0
last_kmer = 1000
first_attention=0
last_attention=1000
mem_length = 1024

print(len(labels[curr_sample]))

y_ss_indicators = ["SS" if labels[curr_sample][i]==1 else '' for i in range(first_kmer,last_kmer)]
x_ss_indicators = ["SS" if i>=0 and labels[curr_sample][i]==1 else '' for i in range(first_attention-1024,last_attention-1024)]

#list of attentions per window
attention_weights = get_attentions_for_sample(gene_ids[curr_sample])
#just those in the ith window
window_attention_weights = attention_weights[0]
attention_to_plot= window_attention_weights[first_kmer:last_kmer,first_attention:last_attention]
print(np.shape(attention_to_plot))

fig, ax = plt.subplots()
im = ax.imshow(attention_to_plot)
im, cbar = heatmap(attention_to_plot,y_ss_indicators,x_ss_indicators,ax=ax,
                   cmap="GnBu", cbarlabel="attention")
#annotate_heatmap(im, valfmt="{x:.1f} t")

#fig.tight_layout()
#plt.savefig('correlation_plot.png', format="png")
fig.set_size_inches(10.5, 10.5)
plt.show()








'''
window_number = 2
kmer_range = 150
splice_sites = np.where(np.array(labels) ==1)
selected_splice_site = splice_sites[0][0]
print(splice_sites)
print(selected_splice_site)
min_kmer = np.maximum(0, selected_splice_site - kmer_range)
max_kmer = np.minimum(1012, selected_splice_site + kmer_range)
print(min_kmer)
print(max_kmer)


input_tensor = torch.tensor(gene_ids).unsqueeze(0).long()
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
'''