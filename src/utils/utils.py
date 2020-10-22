# built-in packages
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
# my custom packages
from src.utils.vocabulary import Vocabulary


def set_cuda(config, n_cuda=None):
    """
    This function is to check cuda available and set device.

    @Arguments:
        config: custom config setting
        n_cuda (string): set specific single gpu (e.g. "0")
    """
    if n_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = n_cuda
    else:
        config.use_cuda = config.use_cuda and torch.cuda.is_available()
        print("cuda on: ", config.use_cuda)
        if config.use_cuda:
            torch.cuda.manual_seed(config.set_seed)
            torch.backends.cudnn.deterministic = True
        else:
            torch.manual_seed(config.set_seed)   
        config.device = torch.device("cuda" if config.use_cuda else "cpu")
    return 

def modify_config_type(config):
    config.use_cuda = True if config.use_cuda in ["1", "True"] else False
    config.set_seed = int(config.set_seed)
    config.num_features = int(config.num_features)
    config.n_classes = int(config.n_classes)
    config.test_batch_size = int(config.test_batch_size)
    config.batch_size = int(config.batch_size)
    config.epochs = int(config.epochs)
    config.log_every = int(config.log_every)
    config.drop_out = float(config.drop_out)
    config.lr = float(config.lr)
    config.lr_decay = float(config.lr_decay)
    config.lr_min = float(config.lr_min)
    config.n_bad_loss = float(config.n_bad_loss)
    config.num_words = int(config.num_words)
    return config
    
def seed_torch(seed=0):
    """
    This function is to set seed for all randomization.

    @Arguments:
        seed (int): the seed index
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def read_json_files(folder_path, n=None):
    """
    This function is to read multiple json files under a specific folder.

    @Arguments:
        folder_path (string): path to the folder which contains json files
        n (int): the number of files (order by the file name), which is set to None if you want to read all
    @Return:
        df (pd.DataFrame): a concatenated DataFrame with all json data
    """
    print("read the datasets...")
    files = []
    file_path = [file for file in os.listdir(folder_path) if file.endswith(".json")]
    # file_path = ['Video_Games_5.json', 'Musical_Instruments_5.json']
    for file in file_path:    
        files.append(pd.read_json(os.path.join(folder_path, file), lines=False))
    df = pd.concat(files)
    return df

def token_statistics(tokens, message=None):
    """
    This function is to print the statistics of token length

    @Arguments:
        tokens (list of object): a list of tokens
    @Return:
        None
    """
    print("max:", max(tokens))
    print("min:", min(tokens))
    print("median:", np.median(tokens))
    print("mean:", np.mean(tokens))
    print("Quantile (25%):", np.quantile(tokens, 0.25))
    print("Quantile (75%):", np.quantile(tokens, 0.75))
    print("Quantile (90%):", np.quantile(tokens, 0.9))
    return

def reclassify(labels):
    """
    This function is to transform the target from 5 classes to binary classes
    1. score 3 & 4 & 5 -> 1 (positive)
    2. score 1 & 2 -> 0 (negative)

    @Arguments:
        labels (list of int): the target variable
    @Return:
        targets (list of int): the transformed target variable
    """
    targets = []
    for gp in labels:
        if gp in [3, 4, 5]:
            targets.append(1)
        elif gp in [1, 2]:
            targets.append(0)
    assert len(targets) == len(labels)
    return targets

def dataset_split(df, train_percent, val_percent, set_seed):
    """
    This function is to split the dataframe into train, validation, and test sets.

    @Arguments:
        df (pd.DataFrame): the original dataset
        train_percent (float): the sample percentage for the training set among the original dataset
        val_percent (float): the sample percentage for the validation set among the original dataset
        set_seed (int): a seed for any randomization for reproducing the result
    @Return:
        train_df (pd.DataFrame): the training data
        val_df (pd.DataFrame): the validation data
        test_df (pd.DataFrame): the testing data
    """
    n = df.shape[0] # get length of dataframe

    # I will set the percentage of validation and test sets to be both 0.1
    train_index, rest_index = train_test_split(range(n), train_size=train_percent, random_state=set_seed)
    val_index, test_index = train_test_split(rest_index, train_size=(val_percent/(1-train_percent)), random_state=set_seed) # 0.1/(1-0.8) = 0.1/0.2 = 0.5

    # check if there is any intersection among all three sets
    assert len(set(train_index + val_index + test_index)) == n

    # get sub_datasets by random_split
    np.random.seed(set_seed)
    train_df, val_df, test_df = df.iloc[train_index, :].reset_index(drop=True),\
                                df.iloc[val_index, :].reset_index(drop=True),\
                                df.iloc[test_index, :].reset_index(drop=True)

    # check random_split works correctly
    assert (len(train_df)+len(val_df)+len(test_df)) == n
    return train_df, val_df, test_df

def build_vocabulary(tokens):
    """
    This function is to build a vocabulary list from a list of tokens.

    @Arguments:
        tokens (list of string): a list of tokens
    @Return
        vocabulary (class object): the Vocabulary object
    """
    vocabulary = Vocabulary()
    for row in tokens:
        for word in row:
            vocabulary.add_word(word)
    return vocabulary

def undersampling(df, target, set_seed=123):
    """
    This function is to do undersampling on the majority target group.

    @Arguments:
        df (pd.DataFrame): the original dataset which will be done for undersampling
        target (string): the name of the target variable
        set_seed (int): a seed for any randomization for reproducing the result
    @Return:
        df (pd.DataFrame): the dataset after doing undersampling
    """
    np.random.seed(set_seed)
    df = df.groupby(target)
    df = df.apply(lambda x: x.sample(df.size().min())).sample(frac=1).reset_index(drop=True)
    return df