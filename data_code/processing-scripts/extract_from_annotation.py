#extracts protein coding genes start and end
#for each protein coding gene, finds all the ss
#doing this because files downloaded from biomart are unreliable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#gene is key, has a tuple of gene name,chr, start, end, and a set of ss
gene_dict ={}



def add_to_gene_dict(entry):
    chrom = entry[1]
    start = int(entry[3])
    end = int(entry[4])
    strand=entry[5]
    desc = entry[6].split(";")
    gene_id = desc[0][8:].strip("\"")
    gene_name = desc[2][11:].strip("\"")
    gene_dict[gene_id]=(gene_name,chrom,strand,start,end,set())

#really adding splice sites not exons to set
def add_exon_to_set(entry):
    exon_start = int(entry[3])
    exon_end = int(entry[4])
    desc = entry[6].split(";")
    gene_id = desc[0][8:].strip("\"")
    exon_set = gene_dict[gene_id][5]
    gene_start = gene_dict[gene_id][3]
    gene_end = gene_dict[gene_id][4]
    #want to somewhat arbitrarily gate this difference to be greater than 50 bp or so so were not mixing it up with alternative transcription start sites/ TTS
    if(exon_start-gene_start>50):
        exon_set.add((exon_start-1))
    if(gene_end-exon_end>50):
        exon_set.add((exon_end+1))

#this is done as a post-processing step so that we can also remove genes with only one exon.
def filter_genes(max_length,min_length=0):
    print("Filtering Out Genes with  %s < Length < %s and only 1 exon" % (max_length,min_length))
    genes_to_remove = []
    for gene_id in gene_dict:
        gene_tuple = gene_dict[gene_id]
        length = gene_tuple[4]-gene_tuple[3]
        if(length>max_length or length <min_length):
            genes_to_remove.append(gene_id)
            continue
        elif(len(gene_tuple[5])<2):
            genes_to_remove.append(gene_id)
            continue
    #Actually go through the list and remove them
    for gene_id in genes_to_remove:
        del gene_dict[gene_id]

def compute_summary_stats():
    print("Computing Summary Stats")
    num_genes = len(gene_dict.keys())
    lengths = []
    num_ss_list=[]
    for gene_id in gene_dict.keys():
        gene_tuple = gene_dict[gene_id]
        lengths.append(gene_tuple[4]-gene_tuple[3])
        num_ss_list.append(len(gene_tuple[5]))
    print("Number of genes: ", num_genes)
    print("Average Gene Length: ",sum(lengths)/num_genes)
    print("Average Number of Splice Sites Per Gene: ", sum(num_ss_list)/num_genes)
    print(sum(num_ss_list)/sum(lengths))
    plot_things(lengths,num_ss_list)

def plot_things(gene_lengths,num_ss_per_gene):
    fig, axs = plt.subplots(ncols=2)
    ss_freq = np.array(num_ss_per_gene)/np.array(gene_lengths)
    g1=sns.distplot(num_ss_per_gene,kde=False,norm_hist=False,ax=axs[0])
    g2=sns.distplot(gene_lengths,kde=False,norm_hist=False,ax=axs[1])
    g1.set_title("Splice Site Frequency")
    g2.set_title("Gene Length")
    g1.set_xlabel("SS Frequency (ss/bp)")
    g2.set_xlabel("Gene Length")
    g1.set_ylabel("Count")
    g2.set_ylabel("Count")
    fig.tight_layout()
    #g.set_xscale('log')
    '''
    fig,axs = plt.subplots(1,3,figsize=(15,3),tight_layout=True)
    axs[2].set_title("Log-Transformed Gene Length Distribution")
    axs[2].set_xscale("log")
    axs[2].hist(gene_lengths,bins=100)
    axs[2].set_xlabel("Log(Gene Length)")
    axs[2].set_ylabel("Counts")
    axs[1].set_title("Gene Length Distribution")
    axs[1].hist(gene_lengths,bins=100)
    axs[1].set_xlabel("Gene Length")
    axs[1].set_ylabel("Counts")
    axs[0].set_title("Splice Site Per Gene")
    axs[0].hist(num_ss_per_gene,bins=100)
    axs[0].set_xlabel("Splice Sites")
    axs[0].set_ylabel("Counts")
    '''
    plt.show()


def write_output():
    with open("../output/100k_gene_ss_info.txt", 'w') as output:
        output.write("Gene ID\tGene Name\tChrom\tStrand\tStart\tEnd\tRelative Splice Site Pos\n")
        for gene_id in gene_dict:
            gene_tuple = gene_dict[gene_id]
            ss_positions = str(sorted(list(gene_tuple[5])))
            output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(gene_id,gene_tuple[0],gene_tuple[1],gene_tuple[2],gene_tuple[3],gene_tuple[4],ss_positions))



def fill_gene_dict():
    df = pd.read_csv('../gencode.v26.annotation.gtf',skiprows=5, sep="\t", header=None,usecols=[0,2,3,4,6,8])
    print("Read in File")
    for entry in df.itertuples():
        level = entry[2]
        desc = entry[6].split(";")
        gene_type = desc[1][11:].strip("\"") if level == "gene" else desc[2][11:].strip("\"")
        if(gene_type=="protein_coding"):
            if level=="gene":
                add_to_gene_dict(entry)
            elif level=="exon":
                add_exon_to_set(entry)
    print("Finished filling gene dictionary")



fill_gene_dict()
#print("Before Filtering")
#compute_summary_stats()
filter_genes(100000)
compute_summary_stats()
#print("After Filtering")
#compute_summary_stats()
write_output()

#50k:12292 genes at 19k avg length
#100k: 15393 genes at 29k avg length