import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../labs_brown/fairbrother_lab')
from utilities import get_bp_in_interval, reverse_complement


#add up to k-1 bp to make the sequence divisible
def get_length_adjusted_sequence(chrom,strand,start,end,k):
    length = end-start+1
    #modify to add 0 if mod is 0
    bp_to_add = k - length%k
    if(strand=="+"):
        raw_seq=get_bp_in_interval(chrom,start,end+bp_to_add)
    elif(strand=="-"):
        raw_seq=reverse_complement(get_bp_in_interval(chrom,start-bp_to_add,end))
    else:
        raise ValueError("Received strand value other than + or -")


def split_into_kmers(seq,seq_length,k):
    kmer_list=[ seq[i:i+k] for i in range(0, len(seq), k) ]
    if(len(kmer_list[-1])!=k):
        raise Exception("Something off in length normalization, kmer not of length k observed")
    return kmer_list

def run(k,gene_ss_df):
    print("Creating input samples. K-mer size = "+str(k))
    processed_genes = []
    for entry in gene_ss_df.itertuples():
        gene_id = entry[1]
        gene_chrom = entry[3]
        gene_start = int(entry[4])
        gene_end = int(entry[5])
        raw_seq, seq_length = get_length_adjusted_sequence(gene_chrom,gene_start,gene_end,k)
        kmers=split_into_kmers(raw_seq,seq_length,k)
        processed_genes.append((gene_id,kmers))
    return processed_genes

def write_samples(samples,k):
    print("writing to file")
    df = pd.DataFrame(samples, columns =['Gene', 'Kmers'])
    df.to_csv(path_or_buf="../output/%s-mer_input_sequences.txt"%k,sep="\t",index=False)

k=6
processed_genes=run(k,pd.read_csv('../output/100k_gene_ss_info.txt', sep="\t", header=0))
write_samples(processed_genes,k)