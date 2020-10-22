# Use the below script at line 2 to run this file
# PYTHONPATH="/scratch/kll482/cathay:$PYTHONPATH" python training.py

import os
import pickle
import torch
from configparser import ConfigParser
import argparse
from src.train_test.training_process import training_process
from src.models.gcn_conv import BinGCNConv
from src.models.gat_conv import BinGATConv
from src.utils.utils import modify_config_type
from src.utils.utils import set_cuda

if __name__ == "__main__":
    # 1. Setting
    print("1. Config Setting")
    myconfig = ConfigParser()
    myconfig.read("config/config.ini")
    config = argparse.Namespace(**myconfig["graph_models"])

    # 2. Modify data type for the config settings
    config = modify_config_type(config)
    
    # 3. Use cuda
    set_cuda(config)

    # 4. Start training the model
    print("Which model I am training? {} \n".format(config.model_name))
    if config.model_name == "gcn":
        model = BinGCNConv(config, config.num_features, config.n_classes, drop=config.drop_out).to(config.device)
    elif config.model_name == "gat":
        model = BinGATConv(config, config.num_features, config.n_classes, drop=config.drop_out).to(config.device)
        
    training_process(config=config, model=model)