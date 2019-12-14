


#TODO: Fix this so it doesnt create the kmers for every tissue but only once

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../../labs_brown/fairbrother_lab')
from utilities import get_bp_in_interval, reverse_complement


def read_in_gene_pos():
    dic={}
    df= pd.read_csv(genes_to_reference, sep="\t", header=0,usecols=[0,2,3,4,5])
    for entry in df.itertuples():
        dic[entry[1]]=(entry[2],entry[3],int(entry[4]),int(entry[5]))
    return dic

#just reading it in, not doing any processing on it
def read_in_tissue_splice_data(tissue_file):
    dic ={}
    df= pd.read_csv(tissue_file, sep="\t", header=0)
    for entry in df.itertuples():
        gene = entry[1]
        sites=np.array(eval(entry[3]))
        reads=np.array(eval(entry[4]))
        #if theres actually data for this gene in GTEx
        if(len(sites)!=0):
            dic[gene]=(sites,reads)
    return dic


#add up to k-1 bp to make the sequence divisible
def get_length_adjusted_sequence(chrom,strand,start,end):
    length = end-start+1
    #modify to add 0 if mod is 0
    bp_to_add = k - length%k
    if(strand=="+"):
        raw_seq=get_bp_in_interval(chrom,start,end+bp_to_add)
    elif(strand=="-"):
        raw_seq=reverse_complement(get_bp_in_interval(chrom,start-bp_to_add,end))
    else:
        raise ValueError("Received strand value other than + or -")
    return raw_seq, length+bp_to_add

def split_into_kmers(seq,seq_length):
    kmer_list=[ seq[i:i+k] for i in range(0, len(seq), k) ]
    if(len(kmer_list[-1])!=k):
        raise Exception("Something off in length normalization, kmer not of length k observed")
    return kmer_list

def create_labels(gene_start,gene_end,strand,seq_positions,normalized_counts):
    gene_length = gene_end-gene_start+1
    adjusted_length = gene_length +k - gene_length%k
    num_kmers=adjusted_length//k
    labels = np.zeros(num_kmers)
    if(strand=="+"):
        for i in range(len(seq_positions)):
            ss=int(seq_positions[i])
            splicing_count=float(normalized_counts[i])
            offset_from_tss=ss-gene_start
            bin_number = offset_from_tss//k
            labels[bin_number]=splicing_count
    elif(strand=="-"):
        for i in range(len(seq_positions)):
            ss=int(seq_positions[i])
            splicing_count=float(normalized_counts[i])
            offset_from_tss=gene_end-ss
            bin_number = offset_from_tss//k
            labels[bin_number]=splicing_count
    else:
        raise ValueError("Received strand value other than + or -")
    return list(labels)

def process_genes(tissue_splicing_data):
    print("Processing genes. K-mer size = "+str(k))
    processed_samples = []
    for gene in tissue_splicing_data:
        pos=gene_pos_dict[gene]
        gene_chrom = pos[0]
        gene_strand=pos[1]
        gene_start = pos[2]
        gene_end = pos[3]
        ss_positions, ss_reads = tissue_splicing_data[gene]
        raw_seq, seq_length = get_length_adjusted_sequence(gene_chrom,gene_strand,gene_start,gene_end)
        if('N' in raw_seq):
            print("gene removed because of N")
            continue
        kmers=split_into_kmers(raw_seq,seq_length)
        normalized_counts=normalize_counts_per_gene(ss_reads)
        labels = create_labels(gene_start,gene_end,gene_strand,ss_positions,normalized_counts)
        processed_samples.append((gene,kmers,labels))
    return processed_samples

#takes in a numpy array and returns a numpy array
def normalize_counts_per_gene(counts_in_gene):
    max_val = max(counts_in_gene)
    if(max_val>0):
        return np.divide(counts_in_gene,max_val)
    else:
        return counts_in_gene

#this method splits genes into regions of 1k bins just so the samples become smaller
#returns a list of tuples (gene id, split #, kmers, labels)
def split_into_small_samples(samples_per_gene):
    size_to_split_into = 1000
    print("Splitting into samples of size",size_to_split_into)
    split_samples = []
    #not adding any padding as this is done in model code
    for entry in samples_per_gene:
        gene_id = entry[0]
        kmers = entry[1]
        labels = entry[2]
        kmer_chunks = [kmers[size_to_split_into*i:size_to_split_into*(i+1)] for i in range(len(kmers)//size_to_split_into + 1)]
        label_chunks = [labels[size_to_split_into*i:size_to_split_into*(i+1)] for i in range(len(kmers)//size_to_split_into + 1)]
        assert len(kmer_chunks)==len(label_chunks)
        for i in range(len(kmer_chunks)):
            split_samples.append((gene_id,i,kmer_chunks[i],label_chunks[i]))
    return split_samples

def write_whole_samples(tissue,samples):
    print("writing to file: ",tissue)
    df = pd.DataFrame(samples, columns =['Gene', 'Sample', 'Labels'])
    df.to_csv(path_or_buf="./output/{}-mer_whole_samples/{}.txt".format(k,tissue),sep="\t",index=False)

        
def write_split_samples(tissue,split_samples):
    print("writing split samples to file:", tissue)
    df = pd.DataFrame(split_samples, columns =['Gene','Gene Part', 'Sample', 'Labels'])
    df.to_csv(path_or_buf="./output/{}-mer_split_samples/{}.txt".format(k,tissue),sep="\t",index=False)



def spot_check(samples):
    for i in range(5):
        gene_data = samples[i]
        gene=gene_data[0]
        one_labels=gene_data[2]
        kmers = gene_data[1]
        ss_indices=[j for j in range(len(one_labels)-1) if one_labels[j]!=0]
        print(gene)
        print(ss_indices)
        ss_kmers=[kmers[j]for j in ss_indices]
        print(ss_kmers)

def run():
    #for every tissue file in the directory, execute this script
    input_dir="./output/ss_reads_by_tissue/"
    for entry in os.listdir(input_dir):
        #ignore hidden files
        #if(entry[0]!="."):
        #only do this for Thyroid
        if(entry[0]=="T"):
            tissue = entry[:-4]
            print(tissue)
            tissue_splicing_data = read_in_tissue_splice_data(input_dir+entry)
            processed_genes = process_genes(tissue_splicing_data)
            spot_check(processed_genes)
            write_whole_samples(tissue,processed_genes)
            #write_split_samples(tissue,split_into_small_samples(processed_genes))



#GLOBALS
#eventually get these from command line
k=2
genes_to_reference="../output/100k_gene_ss_info.txt"
#keyed by gene id to a tuple of (chr, start, end)
gene_pos_dict=read_in_gene_pos()
run()



