import numpy as np
import pandas as pd
from math import ceil
from time import time

#need a dictionary of sample id to tissue and to total number of mapped reads (for normalization)
#will need to normalize junction counts with total number of reads mapped (SMMPPD)
#actually we probably just want to normalize by gene. We don't care about gene expression, just ss usage within gene (conditioned on gene)


#given the sampID dict and the header row of the splice junction file, count how many times each tissue type is observed.
#also just prints stats
def count_samples_per_tissue(sampID_dict):
    print("Counting samples per tissue")
    sample_ids=pd.read_csv(junction_counts_file,sep="\t",skiprows=2, nrows=0).columns.tolist()[2:]
    tissue_counts={}
    for sampID in sample_ids:
        tissue = sampID_dict[sampID][0]
        if tissue not in tissue_counts:
            tissue_counts[tissue]=1
        else:
            tissue_counts[tissue]+=1

    
    filtered_tissue_counts ={}
    for tissue in tissue_counts:
        '''
        #only want to keep tissues with more than 50 samples
        if tissue_counts[tissue]>50:
            filtered_tissue_counts[tissue]=tissue_counts[tissue]
        '''
        #only want to look at Brain, Adipose, and Thyroid. These have low variance
        if(tissue in ["Brain","Adipose Tissue","Thyroid"]):
            filtered_tissue_counts[tissue]=tissue_counts[tissue]
    print("Tissues to be computed:")
    print(filtered_tissue_counts)

    return filtered_tissue_counts

#in order to know what tissue each sample is from and what value to normalize it with
#operates under the assumption that the order of the samples is in the same order as the header, which must be true.
def create_augmented_header(sampID_dict):
    print("Creating header to facilitate parsing")
    #want to exclude the first 2 columns which are pos and gene
    sample_ids=pd.read_csv(junction_counts_file,sep="\t",skiprows=2, nrows=0).columns.tolist()[2:]
    return [sampID_dict[entry] if (entry in sampID_dict and (sampID_dict[entry][0] in counts_per_tissue)) else None for entry in sample_ids]


#will return a dict of sample id to tissue type and intregenic reads.
def read_in_sample_attributes():
    print("Reading in GTEx sample annotations")
    sampID_df = pd.read_csv(sample_annotation_file, sep="\t",usecols=[0,5,33])
    #df of sample ID, tissue type, and intragenic reads (used for normalization)
    sampID_dict = {}
    for entry in sampID_df.itertuples():
        #exclude those that don't provide a total reads mapped (i.e. are not TruSeq)
        if not np.isnan(entry[3]):
            sampID_dict[entry[1]]=(entry[2],entry[3])
    return sampID_dict


def read_in_genes_to_consider():
    df = pd.read_csv('../output/100k_gene_ss_info.txt', sep = "\t",usecols=[0,3])
    return df.set_index('Gene ID')['Strand'].to_dict()



def parse_ss_reads_per_tissue(genes_of_interest):
    chunk_size=5000
    #create and populate dict with tissue types
    #dict inside a dict inside a dict
    #tissue 1:
        #gene 1:
            #ss 1: count
            #ss 2: count
        #gene 2:
        #...
    dict_of_ss_counts_per_tissue={}

    #list of all the tissue types we are considering
    tissue_types = [*counts_per_tissue]
    for tissue in tissue_types:
        dict_of_ss_counts_per_tissue[tissue]={}
        #add dicts for all the genes we are interested in
        for gene in genes_of_interest:
            dict_of_ss_counts_per_tissue[tissue][gene]={}
    
    intra_tissue_stats_per_junction={}

    chunk_num=0
    for chunk in pd.read_csv(junction_counts_file, sep="\t",skiprows=2,chunksize=chunk_size):
        chunk_num+=1
        print("Starting chunk "+str(chunk_num)+" of " +str(ceil(357749/chunk_size)))
        start_time=time()
        for entry in chunk.itertuples():
            junction = entry[1].split("_")
            gene = entry[2]
            if(gene not in genes_of_interest):
                continue
            #toggle between computing variance and actually totalling counts here
            aggregate_ss_gene_over_tissues(entry[3:],gene,junction,dict_of_ss_counts_per_tissue)
            intra_tissue_stats_per_junction[(entry[1],gene)]=compute_intra_tissue_stats(entry[3:],gene,junction)
        print("Time to process chunk: "+str(time()-start_time))
    return dict_of_ss_counts_per_tissue,intra_tissue_stats_per_junction


#this takes as input the dictionary output by parse_ss_reads_per_tissue and 
def write_ss_pos_and_counts_by_tissue(dict_by_tissue,gene_strand_dict):
    print("Writing counts to files. Will produce one per tissue")
    for tissue in dict_by_tissue:
        gene_ss_counts_dict = dict_by_tissue[tissue]
        with open("output/ss_reads_by_tissue/"+tissue+".txt","w") as f:
            f.write("Gene\tStrand\tSplice Sites\tRead Counts\n")
            for gene in gene_ss_counts_dict:
                counts_per_ss= gene_ss_counts_dict[gene]
                strand = gene_strand_dict[gene]
                ss_list = []
                counts_list = []
                for ss in counts_per_ss:
                    ss_list.append(int(ss))
                    counts_list.append(counts_per_ss[ss])
                f.write("{}\t{}\t{}\t{}\n".format(gene,strand,ss_list,counts_list))
    print("Finished writing tissue counts files")


#total over tissues for one row
#returns a total count, not an average, not sure if this is a good thing?
def aggregate_ss_gene_over_tissues(row,gene,junction,dict_of_ss_counts_per_tissue):
    #len(row)=17382
    for i in range(17382):
        count = row[i]
        #get the tissue associated with this position
        tissue,total_reads = samples_header[i] if samples_header[i]!=None else (None,None)
        if tissue==None:
            #then this is one of the tissues that we are ignoring because there arent enough samples.
            continue
        rpm = compute_reads_per_million(count,total_reads)
        gene_junction_sub_dict=dict_of_ss_counts_per_tissue[tissue][gene]
        #add the junction read counts to the dictionary
        gene_junction_sub_dict[junction[1]]= gene_junction_sub_dict[junction[1]]+rpm if junction[1] in gene_junction_sub_dict else rpm
        gene_junction_sub_dict[junction[2]]= gene_junction_sub_dict[junction[2]]+rpm if junction[2] in gene_junction_sub_dict else rpm

#takes in a row (one splice junction) and computes variance for it within each tissue type
def compute_intra_tissue_stats(row,gene,junction):
    #Premature optimization is the root of all evil!

    #allows me to index into the dict to the correct position
    tissue_reads_dict,num_times_tissue_seen_dict = get_empty_array_of_correct_size()
    for i in range(0,len(row)):
        count = row[i]
        #get the tissue associated with this position
        tissue, total_reads = samples_header[i] if samples_header[i]!=None else (None,None)
        if tissue==None:
            #then this is one of the tissues that we are ignoring because there arent enough samples.
            continue
        rpm = compute_reads_per_million(count,total_reads)

        num_times_tissue_seen=num_times_tissue_seen_dict[tissue]
        tissue_reads_dict[tissue][num_times_tissue_seen]=rpm
        num_times_tissue_seen_dict[tissue]+=1
    tissue_variance_dict={}
    for tissue in tissue_reads_dict:
        var = np.var(tissue_reads_dict[tissue])
        mean = np.mean(tissue_reads_dict[tissue])
        tissue_variance_dict[tissue]=(mean,var)
    return tissue_variance_dict

def write_junction_stats(stats_dict_by_junction):
    print("Writing junction stats file.")
    tissue_types=[*counts_per_tissue]
    header=["{}Mean\t{}Variance".format(tissue,tissue) for tissue in tissue_types]
    with open("output/junction_stats.txt","w") as f:
        f.write("Junction\tGene\t"+"\t".join(header)+"\n")
        for pair in stats_dict_by_junction:
            stats = stats_dict_by_junction[pair]
            row = ["{}\t{}".format(str(stats[tissue][0]),str(stats[tissue][1])) for tissue in tissue_types]
            f.write("{}\t{}\t".format(pair[0],pair[1])+"\t".join(row)+"\n")




#Helper function for computing variance
#Since we know the sizes of the array we need we can just pre-create them with the correct size.
#we know the size ahead of time because the number of columns corresponding to each tissue type is constant.
def get_empty_array_of_correct_size():
    empty_dict={}
    num_times_tissue_seen={}
    for tissue in counts_per_tissue:
            empty_dict[tissue]=np.empty(counts_per_tissue[tissue])
            num_times_tissue_seen[tissue]=0
    return empty_dict,num_times_tissue_seen


def compute_reads_per_million(counts,total_number_of_reads_for_sample):
    multiplier = 1000000/total_number_of_reads_for_sample
    return counts*multiplier


#Globals#########
sample_annotation_file = 'individual_sample_data/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
junction_counts_file = 'individual_sample_data/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct'
samples_header=[]
counts_per_tissue={}
#################

def run():
    global samples_header
    global counts_per_tissue
    gene_strand_dict= read_in_genes_to_consider()
    genes=list(gene_strand_dict.keys())
    sampID_dict=read_in_sample_attributes()
    counts_per_tissue=count_samples_per_tissue(sampID_dict)
    samples_header=create_augmented_header(sampID_dict)
    print("Starting to parse junction file. This will take a while...")
    filled_dictionary,filled_stats = parse_ss_reads_per_tissue(genes)
    write_ss_pos_and_counts_by_tissue(filled_dictionary,gene_strand_dict)
    write_junction_stats(filled_stats)

run()