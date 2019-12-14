#3k test
#3k val
#rest training
#Making test and validation so big because we probably cant train on all the genes anyway
import numpy as np
import pandas as pd
import sys
import os
from sklearn.utils import shuffle
np.random.seed(420)
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
TRAIN_SPLIT = 1- TEST_SPLIT-VAL_SPLIT

def parse_input():
    instructions = "Please invoke this script with one argument: The samples file location"
    if(len(sys.argv)!=2):
        print(instructions)
        sys.exit()
    else:
        return sys.argv[1]

def do_split(entire_df):
    print("Splitting dataset")
    num_rows = len(entire_df.index)
    print("Total samples:", num_rows)
    train_num = int(num_rows*TRAIN_SPLIT)
    val_num = int(num_rows*VAL_SPLIT)
    shuffled_df=shuffle(entire_df)
    train_df = shuffled_df[:train_num]
    val_df = shuffled_df[train_num:train_num+val_num]
    test_df = shuffled_df[train_num+val_num:]
    return train_df, val_df, test_df


def write_part(split_part,sub_df,sub_dir_path,data_kind):
    print("writing {} split".format(split_part))
    print(sub_df.shape)
    sub_df.to_csv("{}/{}_{}.txt".format(sub_dir_path,data_kind,split_part),sep="\t",index=False)



def run():
    file_path = parse_input()
    #data_kind = file_path.split("/")[-1][:-4]
    data_kind = "Thyroid"
    entire_df = pd.read_csv(file_path,sep="\t")
    train, val, test = do_split(entire_df)
    sub_dir_path="../output/partitioned_samples/{}".format(data_kind)
    try:
        os.makedirs(sub_dir_path)
    except FileExistsError:
        print("Not creating directory {} because it already exists".format(sub_dir_path))
    write_part("train",train,sub_dir_path,data_kind)
    write_part("val",val,sub_dir_path,data_kind)
    write_part("test",test,sub_dir_path,data_kind)

run()
