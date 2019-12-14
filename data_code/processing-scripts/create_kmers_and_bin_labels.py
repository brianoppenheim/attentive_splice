


#this file is intended to split unspliced genes into k-mers and create the 1 or 0 label for each k-mer based on ss data.
#If a gene length is not divisible by k, an extra couple of bp will be added to the end to make it so

#Pipeline:
'''
1. Read in data as pandas df and convert it to list of tuples
2. Look at seq length and add extra bp if needed to make it divisible by k.
3. Obtain sequence of gene from assembly
4. Split into kmers
5. Bin the splice sites into kmers
6. Write the output to a file as three columns: gene_id, list of kmers, list of 0s and 1s
'''



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
    return raw_seq, length+bp_to_add

def split_into_kmers(seq,seq_length,k):
    kmer_list=[ seq[i:i+k] for i in range(0, len(seq), k) ]
    if(len(kmer_list[-1])!=k):
        raise Exception("Something off in length normalization, kmer not of length k observed")
    return kmer_list

def create_binarized_labels(strand,start,end,ss_list,seq_length,k):
    num_kmers=seq_length//k
    labels = np.zeros(num_kmers)
    if(strand=="+"):
        for ss in ss_list:
            ss=int(ss)
            offset_from_tss=ss-start
            bin_number = offset_from_tss//k
            labels[bin_number]=1
    elif(strand=="-"):
        for ss in ss_list:
            ss=int(ss)
            offset_from_tss=end-ss
            bin_number = int(offset_from_tss)//k
            labels[bin_number]=1
    else:
        raise ValueError("Received strand value other than + or -")
    return list(labels)

def run(k,gene_ss_df):
    print("Processing genes. K-mer size = "+str(k))
    processed_samples = []
    for entry in gene_ss_df.itertuples():
        gene_id = entry[1]
        gene_chrom = entry[3]
        gene_strand=entry[4]
        gene_start = int(entry[5])
        gene_end = int(entry[6])
        ss_list = eval(entry[7])
        raw_seq, seq_length = get_length_adjusted_sequence(gene_chrom,gene_strand,gene_start,gene_end,k)
        if('N' in raw_seq):
            print("gene removed because of N")
            continue
        kmers=split_into_kmers(raw_seq,seq_length,k)
        labels = create_binarized_labels(gene_strand,gene_start,gene_end,ss_list,seq_length,k)
        processed_samples.append((gene_id,kmers,labels))
    return processed_samples

def write_samples(samples,k):
    print("writing to file")
    df = pd.DataFrame(samples, columns =['Gene', 'Sample', 'Labels'])
    df.to_csv(path_or_buf="../all_samples_%s-mer.txt"%k,sep="\t",index=False)


#this method splits genes into regions of 1k bins just so the samples become smaller
#returns a list of tuples (gene id, split #, kmers, labels)
def split_into_small_samples(samples_per_gene):
    size_to_split_into = 1000
    print("Splitting into samples of size",size_to_split_into)
    split_samples = []
    for entry in samples_per_gene:
        gene_id = entry[0]
        kmers = entry[1]
        labels = entry[2]
        #add padding to kmers and labels so its a multiple of 1 thousand
        #if theres less than 20 hexamers, left at the end, just exclude it entirely
        num_bins = len(kmers)
        bins_over_cutoff = num_bins%size_to_split_into
        padding_to_add = 1000-bins_over_cutoff
        if bins_over_cutoff>20:
            kmers=kmers+["--PAD--"]*padding_to_add
            labels=labels+[0]*padding_to_add
            num_bins=num_bins+padding_to_add
        else:
            kmers = kmers[:num_bins-bins_over_cutoff]
            labels=labels[:num_bins-bins_over_cutoff]
            num_bins=num_bins-bins_over_cutoff
        for i in range(0,num_bins,size_to_split_into):
            curr_split_index = i//size_to_split_into
            thousand_kmers = kmers[i:i+size_to_split_into]
            thousand_labels = labels[i:i+size_to_split_into]
            thousand_sample = (gene_id,curr_split_index,thousand_kmers,thousand_labels)
            split_samples.append(thousand_sample)
    return split_samples


        

def write_small_samples(split_samples,k):
    print("writing split samples to file")
    df = pd.DataFrame(split_samples, columns =['Gene','Gene Part', 'Sample', 'Labels'])
    df.to_csv(path_or_buf="../split_samples_%s-mer.txt"%k,sep="\t",index=False)

def check_input():
    instructions = "Please run this script using three arguments: \n k-mer size (int)\n whether to use full genes or split into samples (-g or -s)\n what input file to read from (one directory up)"
    if(len(sys.argv)!=4):
        print(instructions)
        sys.exit()
    try:
        k = int(sys.argv[1])
    except:
        print("K-mer size must be an int")
        sys.exit()
    if(sys.argv[2]=="-g"):
        small_samples = False
    elif(sys.argv[2]=="-s"):
        small_samples=True
    else:
        print(instructions)
        sys.exit()
    file_name= sys.argv[3]

    return k,small_samples,file_name

def spot_check(samples):
    for i in range(5):
        gene_data = samples[i]
        gene=gene_data[0]
        one_labels=gene_data[2]
        kmers = gene_data[1]
        one_indices=[j for j in range(len(one_labels)-1) if one_labels[j]==1]
        print(gene)
        print(one_indices)
        one_kmers=[kmers[j]for j in one_indices]
        print(one_kmers)


if __name__ == "__main__":
    k,small_samples,file_name = check_input()
    gene_ss_df=pd.read_csv('../output/'+file_name, sep="\t", header=0)
    processed_by_gene = run(k,gene_ss_df)
    spot_check(processed_by_gene)
    if(small_samples):
        split_samples = split_into_small_samples(processed_by_gene)
        write_small_samples(split_samples,k)
    else:
        write_samples(processed_by_gene,k)
    print("Done")



