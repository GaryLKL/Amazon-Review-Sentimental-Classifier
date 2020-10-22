import torch
from torch.utils.data import Dataset
# my custom packages
from src.utils.pipeline import Pipe
from src.preprocessing.feature_engineering.bert_embedding import BertEmbedding


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
class SequenceDataset(Dataset):
    def __init__(self, config, df, use_cuda, embedd_mode="bert", embeddings):
        self.df = df
        self.use_cuda = use_cuda
        self.embeddings = embeddings
        self.config = config
        assert embedd_mode in ["random", "bert"] # I only provide a text transformed dataset for random and bert embeddings
        if embedd_mode == "random":
            self.df[config.nodes] = [self.words_to_indices(row) for row in self.df[config.nodes]] # transform text to index first to make the model faster but memory costly
        elif embedd_mode == "bert":
            bertembedding = BertEmbedding()
            pipe = Pipe([bertembedding.get_embeddings, torch.tensor])
            self.build_embeddings = pipe.call_pipeline
            
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tt = torch.cuda if self.use_cuda else torch
        x = tt.FloatTensor(self.embeddings.get_embeddings(self.df[self.config.tokens][idx]))
#         x = tt.FloatTensor(self.df[self.config.TOKENS][idx])
        y = tt.FloatTensor([self.df[self.config.target][idx]])
        return x, y