# built-in packages
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset, Dataset, download_url
import torch.optim as optim
from datetime import datetime
import os
# my custom packages
from src.train_test.train_utils import train, validate,\
                                        write_log, write_config
from src.utils.utils import seed_torch


def training_process(config, model):
    neighbors = [int(n.strip()) for n in config.neighbor.split(",")]
    for neighbor in neighbors:
        ''' initials '''
        seed_torch(config.set_seed)
        # config.n_classes = 1
        model_name = config.model_name
        config.edge_index = "edgeIndex{}".format(neighbor)
        
        # reload data loaders
        train_loader = torch.load(os.path.join(config.data_loader_path,
                                               "train_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))
        val_loader = torch.load(os.path.join(config.data_loader_path,
                                             "val_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))
        test_loader = torch.load(os.path.join(config.data_loader_path,
                                              "test_loader_{}_{}.pth".format(config.embedd_method, config.edge_index)))  
        print("size of train loader", len(train_loader))

        ''' initializing the model '''
        # model = BinGCNConv(config.NUM_FEATURES, config.N_CLASSES, drop=0).to(device)
        criterion = nn.BCEWithLogitsLoss().to(config.device)

        optimizer = optim.Adagrad(model.parameters(), lr=config.lr)

        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        config.best_model = os.path.join(config.result_path,
                                         "checkpoint/{}_{}_{}.pth".format(config.edge_index,
                                                                          model_name,
                                                                          current_time)
                                        )
        config.log_file = os.path.join(config.log_path,
                                       "{}_{}_{}.txt".format(config.edge_index,
                                                             model_name,
                                                             current_time)
                                      )
        config.config_saved_path = os.path.join(config.result_path,
                                                "config_saved/{}_{}_{}.pkl".format(config.edge_index,
                                                                                   model_name,
                                                                                   current_time)
                                                )
        ''' start training '''
        if 1 == 1:  # change to True to train
            config.iteration = config.n_total = config.train_loss = config.init_bad_loss = 0
            config.stop = False
            config.best_val_loss = float("inf")
            config.init = datetime.now()
            # config.epochs = 15
            # config.log_every = 500
            # config.n_bad_loss 4

            output_text = "\n=== The nodes within the same graph are connecting with only {} neighbors. Let's start training ===\n".format(neighbor) + write_config(config) + "Start record at {}".format(str(datetime.now()))
            write_log(output_text,
                      config.log_file,
                      "w+")

            for epoch in range(1, config.epochs+1):
                train(config, train_loader, val_loader, model, criterion, optimizer, epoch)
                if config.stop:
                    break
    return