# Use the below script at line 2 to run this file
# PYTHONPATH="/scratch/kll482/cathay:$PYTHONPATH" python src/preprocessing/build_loaders.py

# packages
print("Importing packages...")
# built-in packages
from configparser import ConfigParser
import argparse
import os
import torch
from torch_geometric.data import Data, DataLoader

# my custom packages
from src.utils.utils import seed_torch, read_json_files, token_statistics,\
                        reclassify, dataset_split,\
                        build_vocabulary, undersampling, set_cuda
from src.utils.graph_dataset import GraphDataset, get_datasets
from src.utils.utils import modify_config_type


def main(config):

    # 1. Arguments

    # 2. Read all preprocessed json files.
    print("2. Read all preprocessed json files")
    df = read_json_files(folder_path=config.data_path,
                         n=None)

    # 3. Take a look at the token and unique token length distribution among all reviews
    print("3. Take a look at the token and unique token length distribution among all reviews")
    review_count = [len(row) for row in df["reviewTokens"]]
    unique_count = [len(row) for row in df["uniqueTokens"]]
    token_statistics(tokens=review_count,
                     message="=== Review Count ===\n")
    token_statistics(tokens=unique_count,
                     message="=== Unique Review Count ===\n")

    # 4. Remove those data points which have less than 5 unique tokens (unimplemented yet)
    
    # 5. Reclassify overall score from 1~5 to [0,1], representing negative and positive
    print("5. Reclassify overall score from 1~5 to [0,1], representing negative and positive")
    df["y"] = reclassify(df["overall"])
    config.target = "y"

    # 6. Make sure there is no NA value in the new target column
    print("6. Make sure there is no NA value in the new target column")
    assert sum(df[config.target].isna()) == 0

    # 7. Split the dataframe into train, validation, and test sets
    print("7. Split the dataframe into train, validation, and test sets")
    train_df, val_df, test_df = dataset_split(
        df=df,
        train_percent=0.8,
        val_percent=0.1,
        set_seed=config.set_seed,
    )
    
    # 8. Create a vocabulary from unique tokens
    print("8. Create a vocabulary from unique tokens")
    vocabulary = build_vocabulary(train_df[config.nodes])
    config.num_words = vocabulary.num_words
    
    # 9. Conduct undersampling on the based on the target variable distribution
    print("9. Conduct undersampling on the based on the target variable distribution")
    train_df = undersampling(train_df, config.target)
    print("Length of training set:", train_df.shape[0])
    print("Length of test set:", test_df.shape[0])

    # 10. Build Pytorch dataset
    print("10. Build Pytorch dataset for all neighbors")
    print("11. Build and save Pytorch dataloaders")
    
    neighbors = [int(n.strip()) for n in config.neighbor.split(",")]
    for neighbor in neighbors:
        config.edge_index = "edgeIndex{}".format(neighbor)
        train_dataset, val_dataset, test_dataset = get_datasets(config=config,
                                                                root=config.data_path, # "/home/kll482/cathay/dataset",
                                                                vocabulary=vocabulary,
                                                                embedd_mode=config.embedd_method,
                                                                train_df=train_df,
                                                                val_df=val_df,
                                                                test_df=test_df,)
        # check if the row length remains the same even we transform dataframe into graph dataset
        assert len(train_dataset)+len(val_dataset)+len(test_dataset) == len(train_df)+len(val_df)+len(test_df)

        # 11. Build and save Pytorch dataloaders
        seed_torch(config.set_seed)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

        torch.save(train_loader,
                   os.path.join(config.data_loader_path,
                                "train_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))
        torch.save(val_loader,
                   os.path.join(config.data_loader_path,
                                "val_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))
        torch.save(test_loader,
                   os.path.join(config.data_loader_path,
                                "test_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))
    

if __name__ == "__main__":
    # 1. Setting
    print("1. Config Setting")
    myconfig = ConfigParser()
    myconfig.read("config/config.ini")
    config = argparse.Namespace(**myconfig["graph_models"])
    
    # 2. Data Type for Config
    config = modify_config_type(config)
    
    # 3. Start
    set_cuda(config)    
    main(config)
    
    # 4. Write a new config file since we have added num_words
    for k, v in vars(config).items():
        myconfig["graph_models"][k] = str(v)
    with open('config/myconfig.ini', 'w') as configfile:
        myconfig.write(configfile)