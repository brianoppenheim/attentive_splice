#meant to read the list of genes in the splice junction file and write them back out in a readable form.
#to investigate why only 4k protein coding genes are showing up in the parsed file output.
import numpy as np
import pandas as pd

genes=pd.read_csv('individual_sample_data/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct',sep="\t",skiprows=2, usecols=[1])
print(genes.head(20))
print(genes.shape)
genes.drop_duplicates(keep = 'first', inplace = True)
print(genes.head(20))
print(genes.shape)
genes.to_csv(path_or_buf="output/genes_in_sj.txt",sep="\t",index=False)