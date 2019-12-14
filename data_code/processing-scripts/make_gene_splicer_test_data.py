import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../labs_brown/fairbrother_lab')
from utilities import get_bp_in_interval, reverse_complement


#add up to k-1 bp to make the sequence divisible
def get_length_adjusted_sequence(chrom,strand,start,end):
    if(strand=="+"):
        raw_seq=get_bp_in_interval(chrom,start,end)
    elif(strand=="-"):
        raw_seq=reverse_complement(get_bp_in_interval(chrom,start,end))
    else:
        raise ValueError("Received strand value other than + or -")
    return raw_seq


def offset_ss_list(strand,start,end,ss_list):
    new_ss=[]
    if(strand=="+"):
        for ss in ss_list:
            ss=int(ss)
            offset_from_tss=ss-start
            new_ss.append(offset_from_tss)
    elif(strand=="-"):
        for ss in ss_list:
            ss=int(ss)
            offset_from_tss=end-ss
            new_ss.append(offset_from_tss)
    else:
        raise ValueError("Received strand value other than + or -")
    return new_ss

def run(gene_ss_df,testing_ids):
    print("Processing genes")
    gene_to_seq={}
    gene_to_label={}
    for entry in gene_ss_df.itertuples():
        gene_id = entry[1]
        if(gene_id not in testing_ids):
            continue
        gene_chrom = entry[3]
        gene_strand=entry[4]
        gene_start = int(entry[5])
        gene_end = int(entry[6])
        ss_list = eval(entry[7])
        raw_seq=get_length_adjusted_sequence(gene_chrom,gene_strand,gene_start,gene_end)
        labels = offset_ss_list(gene_strand,gene_start,gene_end,ss_list)
        gene_to_seq[gene_id]=raw_seq
        gene_to_label[gene_id]=labels
    return gene_to_seq,gene_to_label

def write_labels(d):
    print("writing labels")
    with open("gene_splice_targets.tsv","w") as f:
        f.write("Gene\tLabels\n")
        for el in d:
            f.write("{}\t{}\n".format(el,d[el]))
def write_fasta(d):
    print("writing fasta files")
    for el in d:
        with open("gene_splicer_input/"+el+".fa","w") as f:
            f.write(">{}\n".format(el))
            f.write(d[el]+"\n")


if __name__ == "__main__":
    testing_ids=pd.read_csv('../output/partitioned_samples/all_samples_6-mer/all_samples_6-mer_test.txt', sep="\t", header=0,usecols=[0])['Gene'].values.tolist()
    print(testing_ids)
    gene_ss_df = pd.read_csv('../output/100k_gene_ss_info.txt', sep="\t", header=0)
    gene_to_seq, gene_to_label = run(gene_ss_df,testing_ids)
    write_fasta(gene_to_seq)
    write_labels(gene_to_label)
    



