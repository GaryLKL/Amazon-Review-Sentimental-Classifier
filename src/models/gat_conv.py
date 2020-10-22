import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


# ref: https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn
# GATConv for binary classification
class BinGATConv(torch.nn.Module):
    def __init__(self, config, embedding_size, n_classes=1, drop=0, task="graph"):
        super(BinGATConv, self).__init__()
        self.conv1 = GATConv(embedding_size, 128, heads=1, dropout=drop) # if you defined cache=True, the shape of batch must be same!
        self.conv2 = GATConv(128, 64, heads=1, dropout=drop)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        self.linear = nn.Linear(64, 64)
        self.classify = nn.Linear(64, n_classes)
        self.task = task
        self.drop = drop
        self.config = config
        if config.embedd_method == "random":
            # I want to finetune random embeddings. However, for pretrained embeddings, like BERT, I will make it static.
            self.embeddings = nn.Embedding(num_embeddings=config.num_words,
                                           embedding_dim=embedding_size)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.config.embedd_method == "random":
            x = self.embeddings(x)
            
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.ln1(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.ln2(x)
        
        # to make a graph-level classification, I need a output size of (total # of graph in a batch) * n_classes
        # therefore, I take the mean of nodes in one GRAPH to represent it output
        if self.task == 'graph':
            x = global_mean_pool(x, batch)
        else:
            pass
        x = self.linear(x)
        x = self.classify(x) # size: (total # of nodes in a batch) * n_classes
        return x.squeeze(-1) # use nn.BCEWithLogitsLoss during training