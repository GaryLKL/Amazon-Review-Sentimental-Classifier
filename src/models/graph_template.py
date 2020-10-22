# https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn
# Just a template for multi-classification problem. May require some modification
class MultiClassNetTemplate(torch.nn.Module):
    def __init__(self, embedding_size, n_classes, drop=0, task="graph"):
        super(MultiClassNet, self).__init__()
        self.conv1 = GCNConv(embedding_size, 64, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.conv2 = GCNConv(64, 32, cached=False)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 32)
        self.classify = nn.Linear(32, n_classes)
        self.task = task
        self.drop = drop
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
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
        x = self.linear(x)
        x = self.classify(x) # size: (total # of nodes in a batch) * n_classes

        return F.log_softmax(x, dim=1)