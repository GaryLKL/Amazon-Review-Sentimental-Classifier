# built-in packages
import os
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
# my custom packages


def train(config, train_loader, val_loader, model, criterion, optimizer, epoch):
    # stop = config.stop.copy()

    logs = "=> EPOCH {}".format(epoch)
    write_log(logs, config.log_file, "a+")
    
    for batch_index, batch in tqdm(enumerate(train_loader)):
        config.iteration += 1 # total iteration within all batches
        
        batch = batch.to(config.device)
        
        if config.n_classes == 1:
            # binary
            label = batch.y.float()
        elif config.n_classes > 1:
            label = batch.y.long()
        label = label.to(config.device).detach()
        
        # train the model
        model.train()
#         for param in model.parameters():
#             print(param.requires_grad)
        output = model(batch)
        
        # loss function
        loss = criterion(output, label)
        
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # sum with the previous training loss for updating learning rate in the following
        config.train_loss += batch.num_graphs * loss.item() # accumulated training loss; batch.num_graphs is the size of the batch
        config.n_total += batch.num_graphs
#         # validation check 
        if config.iteration % config.log_every == 0:
            config.train_loss /= config.n_total 
            val_loss = validate(config, val_loader, model, criterion)
        
            # save logs
            logs = "   % Time: {} | Iteration: {:5} | Batch: {:4}/{}"\
                  " | Train loss: {:.4f} | Val loss: {:.4f}"\
                  .format(str(datetime.now()-config.init), config.iteration, batch_index+1,
                          len(train_loader), config.train_loss, val_loss)
            write_log(logs, config.log_file, "a+")

            # test for val_loss improvement
            config.n_total = config.train_loss = 0
            if val_loss < config.best_val_loss: # update the best validation loss
                config.best_val_loss = val_loss
                config.init_bad_loss = 0
                torch.save(model.state_dict(), config.best_model) # save the checkpoint
            else:
                config.init_bad_loss += 1
            
            # save temporary config setting into a pickle file
            file = open(config.config_saved_path, 'wb')
            pickle.dump(config, file)

            # update the learning rate if val loss does not improve for n_bad_loss times
            if config.init_bad_loss == config.n_bad_loss:
                config.best_val_loss = val_loss
                config.init_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                
                logs = "=> Adjust learning rate to: {}".format(new_lr)
                write_log(logs, config.log_file, "a+")
                
                if new_lr < config.lr_min:
                    config.stop = True
                    break

def validate(config, val_loader, model, criterion):
        
    model.eval()
    val_loss = 0
    dataset_size = 0
    for batch in tqdm(val_loader):
        batch = batch.to(config.device)
        
        if config.n_classes == 1:
            # binary
            label = batch.y.float()
        elif config.n_classes > 1:
            label = batch.y.long()
            
        label = label.to(config.device)
        dataset_size += batch.num_graphs
        
        # train the model
        output = model(batch)
        loss = criterion(output, label)
        val_loss += loss.data * batch.num_graphs
    return val_loss / dataset_size

def test(config, test_loader, model, threshold=0.5):
    print("start testing...")
#     for param in model.parameters():
#         param.requires_grad = False
        
    model.eval()
    dataset_size = 0
    label_list = []
    prediction_list = []
    predict_prob_list = []
    
    for batch in tqdm(test_loader):
        batch = batch.to(config.device)
        
        if config.n_classes > 1:
            label = batch.y.long()
            label = label.data.tolist()
            label_list += label
            output = model(batch)
            _, prediction = torch.max(output, 1)
            prediction = prediction.data.tolist()
            prediction_list += prediction
            
        elif config.n_classes == 1:
            # binary
            label = batch.y.float()            
            label = label.data.tolist()
            label_list += label
            output = model(batch)
            sigmoid = nn.Sigmoid() 
            output = sigmoid(output) # [-inf, inf] -> [0, 1]
            prediction_list += [1 if o > threshold else 0 for o in output.data.tolist()]
            predict_prob_list += output.data.tolist()
        
    confusion_matrix_df = pd.DataFrame(confusion_matrix(label_list, prediction_list))#.rename(columns=["1","2","3","4","5"], index=["1","2","3","4","5"])
    # write the confusion matrix into the log
    write_log("\n{}\n\n{}\n".\
          format("=== Confusion Matrix ===",
                 confusion_matrix_df
                 
                ),
          config.log_file,
          "a+")
    
    # sns.heatmap(confusion_matrix_df, annot=True)
    
    return label_list, prediction_list, predict_prob_list

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

def write_log(text, file_path, mode="a+"):
    """
    This function is to write a log file.

    @Arguments:
        text (string): the text to be written in the log file
        file_path (string): path for the log file
        mode (string): writing mode
    """
    print(text)
    with open(file_path, mode) as file:
        file.write(text+"\n")
    return

def write_config(config):
    """
    This function is to return config setting and variables

    @Arguments:
        config: custom config setting
    @Return:
        texts: texts containing the config setting
    """
    texts = "=== Settings ===\n"
    var_config = vars(config)
    for i in range(len(var_config)):
        temp_text = "{}: {}\n".format(list(var_config.keys())[i],
                                      list(var_config.values())[i],
                                     )
        texts += temp_text
    texts += "============\n\n"
    return texts

def model_load(config, model_test, name):
    if name is None:
        model_reloaded = config.best_model
    else:
        model_reloaded = os.path.join(config.result_path, "checkpoint/{}.pth".format(name))
    model_train = torch.load(model_reloaded)
    model_test.load_state_dict(model_train)

    return model_test