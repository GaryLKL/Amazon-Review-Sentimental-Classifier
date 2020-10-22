# PYTHONPATH="/scratch/kll482/cathay/:$PYTHONPATH" python feature_engineering.py
# import os
# os.chdir("/scratch/kll482/cathay/")
# print("current working directory:", os.getcwd())

# import sys
# sys.path.append("/scratch/kll482/cathay/")

''' packages '''
import os
import json
from configparser import ConfigParser
import argparse
import numpy as np, pandas as pd
import time
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import nltk
from nltk import word_tokenize
import torch
from src.preprocessing.feature_engineering.bert_embedding import BertEmbedding
from src.preprocessing.text_cleaning.text_cleaning import full_step_preprocessing

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

''' read line-delimited json '''
def read_line_json(path, name_list):
    print("start reading the json files...")
    json_contents = []
    for file_name in name_list:
        with open(os.path.join(path, file_name)) as file:
            for i, line in enumerate(file):
                json_dict = json.loads(line)
                json_dict["category"] = file_name[8:-7] # add a column denoting the category
                json_contents.append(json_dict)
    return json_contents

def json_to_df(json_data, selected_cols=None):
    if selected_cols is None:
        data = pd.DataFrame(json_data)
    else:
        data = pd.DataFrame(json_data).loc[:, selected_cols]
    ''' Remove duplicated items if existing... '''
    # data.sort_values('asin').drop_duplicates(subset=['reviewerID','reviewText','unixReviewTime','summary','category'],keep='first',inplace=False)
    ''' Save the DataFrame into a csv file if needed... '''
    # data.to_csv()
    return data

def get_cleaned_tokens(df, review_name="reviewText", review_token_name="reviewTokens", cpu_number=4):
    print("cleaning the reviews...")
    pool = mp.Pool(cpu_number)
    df[review_token_name] = pool.map(full_step_preprocessing, tqdm(df[review_name]))
    pool.close()
    pool.join()
    return df

def remove_empty_tokens(df, review_token_name="reviewTokens"):
    print("remove empty tokens...")
    
    empty_row_index = list(df[review_token_name][df[review_token_name].apply(lambda x: len(x)==0)].index)
    df = df.drop(axis=0, index=empty_row_index).reset_index(drop=True)
    
    assert sum(df[review_token_name].apply(lambda x: len(x)==0)) == 0
    return df


''' get edge index from unique tokens '''
def get_edge_index(tokens, unique_vocabulary, num_neighbor):
    # initialize
#     unique_vocabulary = set(tokens) 
    vocabulary_dict = {value: index for index, value in enumerate(unique_vocabulary)} # dictionary of unique tokens
    edge_start = []
    edge_end = []
    
    # build edge index
    for token_index, token in enumerate(tokens):
        curr_index = vocabulary_dict[token] # current token's index in vocabulary_dict
        
        for p in range(1, num_neighbor+1): # find neighbors of current tokens
            if token_index-p >= 0: # if previous p token exists
                prev_index = vocabulary_dict[tokens[token_index-p]] # get the index of the previous p token
                edge_start += [curr_index, prev_index] # undirected
                edge_end += [prev_index, curr_index]
                
            if token_index+p < len(tokens): # if next p toke exists
                next_index = vocabulary_dict[tokens[token_index+p]] # get the index of the next p token   
                edge_start += [curr_index, next_index]
                edge_end += [next_index, curr_index]
    
    edge_index = [edge_start, edge_end]
    return edge_index

def create_edge_index_col(df, neighbors, review_token_name="reivewTokens",
                          unique_token_name="uniqueTokens"):
    # edge index
    edge_index_names = []
    for neighbor in tqdm(neighbors):
        # get edge index with n neighbors
        edge_index = df.apply(lambda row: get_edge_index(tokens=row[review_token_name],
                                                         unique_vocabulary=row[unique_token_name],
                                                         num_neighbor=neighbor,
                                                        ),
                              axis=1
                             )
        df["edgeIndex{}".format(neighbor)] = edge_index # insert edge indices to the dataframe
        edge_index_names.append("edgeIndex{}".format(neighbor))
    return df, edge_index_names
       

def main(config):
    cpu_count = int(config.cpu_count)
    print("Multiprocessing CPU Count:", cpu_count)
    
    ''' 1. read json '''
    folder_path = config.amazon_file_path
    file_lists = [name for name in os.listdir(folder_path) if name[-5:] == ".json"] # ./amazon_reviews
    # file_lists = [file_lists[0]] # uncomment this line if reading one file
    
    ''' 2. handle and save each file respectively '''
    target = config.target
    review_name = config.review_name
    review_token_name = config.review_token_name
    unique_token_name = config.unique_token_name
    neighbors = [int(n.strip()) for n in config.neighbors.split(",")] # transform neighbors from a string to a list of integers
    is_bidirection = config.is_bidirection
    for idx, file in enumerate(file_lists):
        print("=== preprocessing file {} ===".format(idx))
        
        ''' 2-1. json to dataframe '''
        print("transform json to dataframe...")
        json_contents = read_line_json(folder_path, [file]) # reading the json files
        # cols = ["reviewerID", "asin", "reviewText", "overall", "summary", "unixReviewTime", "category"] # the columns I want to keep:
        df = json_to_df(json_data=json_contents, selected_cols=None)

        ''' 2-2. do text cleaning '''
        print("clean tokens and remove rows with no tokens...")
        df = get_cleaned_tokens(df, review_name, review_token_name, cpu_number=cpu_count)
        df = remove_empty_tokens(df, review_token_name)
    
        ''' 2-3. get unique tokens from review tokens '''
        print("extract unique tokens...")
        df[unique_token_name] = df[review_token_name].apply(lambda row: set(row))
    
        ''' 2-4. get edge index from unique tokens '''
        print("getting the edge index for each graph...")
        df, edge_index_names = create_edge_index_col(df=df,
                                                     neighbors=neighbors,
                                                     review_token_name=review_token_name,
                                                     unique_token_name=unique_token_name)
    
        ''' 2-5. save df as a new json file '''
        df.loc[:, [target, review_token_name, unique_token_name]+edge_index_names] \
            .to_json(os.path.join(config.file_saved_path, "{}".format(file)), orient="columns")
        
        print("finish!")


if __name__ == "__main__":
    myconfig = ConfigParser()
    myconfig.read("config/config.ini")
    config = argparse.Namespace(**myconfig["feature_engineering"])
    main(config)

