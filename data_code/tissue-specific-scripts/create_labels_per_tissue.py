#need to do 2 things:
#1. normalize counts within each gene
#2. bin into kmers from start of gene

import os
import numpy as np
import pandas as pd

def read_in_gene_pos():
    dic={}
    df= pd.read_csv(genes_to_reference, sep="\t", header=0,usecols=[0,2,3,4,5])
    for entry in df.itertuples():
        dic[entry[1]]=(entry[2],entry[3],entry[4],entry[5])
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

#gets the file in form of a dictionary
#loops through all genes and send each one off to processing
#returns a list of tuples (gene,binned_labels)
def process_all_genes_in_tissue(splice_sites_for_tissue):
    list_of_genes_and_labels=[]
    for gene in splice_sites_for_tissue:
        sites = splice_sites_for_tissue[gene][0]
        reads = splice_sites_for_tissue[gene][1]
        list_of_genes_and_labels.append(process_gene_splice_sites(gene,sites,reads))
    return list_of_genes_and_labels


#Processes a single gene
#returns a tuple (gene, binned_labels)
def process_gene_splice_sites(gene_name,sites,reads):
    gene_pos = gene_pos_dict[gene_name]
    strand=gene_pos[1]
    start = gene_pos[2]
    end = gene_pos[3]

    normalized_counts = normalize_counts_per_gene(reads)
    target_labels = bin_into_kmers(start,end,strand,sites,normalized_counts)
    return (gene_name,target_labels)


#takes in a numpy array and returns a numpy array
def normalize_counts_per_gene(counts_in_gene):
    max_val = max(counts_in_gene)
    if(max_val>0):
        return np.divide(counts_in_gene,max_val)
    else:
        return counts_in_gene

def bin_into_kmers(gene_start,gene_end,strand,seq_positions,normalized_counts):
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



def write_processed_tissue_labels(tissue,list_of_labels):
    print("Finished Processing {}. Writing it to File".format(tissue))
    df = pd.DataFrame(list_of_labels, columns =['Gene','Labels'])
    df.to_csv(path_or_buf="./output/{}-mer_labels_by_tissue/{}.txt".format(k,tissue),sep="\t",index=False)


def run():
    #for every tissue file in the directory, execute this script
    input_dir="./output/ss_reads_by_tissue/"
    for entry in os.listdir(input_dir):
        #trying to ignore hidden files
        if(entry[0]!="."):
            tissue = entry[:-4]
            tissue_splicing_data = read_in_tissue_splice_data(input_dir+entry)
            labels=process_all_genes_in_tissue(tissue_splicing_data)
            write_processed_tissue_labels(tissue,labels)


#GLOBALS
#eventually get these from command line
k=6
genes_to_reference="../output/100k_gene_ss_info.txt"
#keyed by gene id to a tuple of (chr, start, end)
gene_pos_dict=read_in_gene_pos()
run()
        