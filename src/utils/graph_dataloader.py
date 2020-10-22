# built-in packages
from torch_geometric.data import Data, DataLoader
# my custom packages
from src.utils.utils import seed_torch


def get_dataloaders(config, dataset, train_percent=0.8, val_percent=0.1):
    '''
    This function will split the full dataset based on given percentage and return the dataloaders for the training, validation, and testing sets.
    '''
    # get length of subsets
    train_len = int(dataset.len()*0.8)
    val_len = int(dataset.len()*0.1)
    test_len = dataset.len()-train_len-val_len
    
    # get sub_datasets by random_split
    seed_torch(config.SEED)
    train_dataset, val_dataset, test_dataset = random_split(dataset, (train_len, val_len, test_len))
    
    # check random_split works correctly
    assert (len(train_dataset)+len(val_dataset)+len(test_dataset)) == dataset.len()
    
    # get dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
    test_loader = DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# example:
# train_loader, val_loader, test_loader = get_dataloaders(config, dataset, train_percent=0.8, val_percent=0.1)