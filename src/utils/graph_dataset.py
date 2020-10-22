# built-in packages
import torch
import torchvision
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset, Dataset, download_url
import torch_geometric.transforms as T
# my custom packages
from src.utils.pipeline import Pipe
from src.preprocessing.feature_engineering.bert_embedding import BertEmbedding


"""
1. Use this class if you don't want to output/save the graph data. Warning: this will read the whole dataframe.
"""
class GraphDataset(Dataset):
    def __init__(self, config, root, df, vocabulary, embedd_mode="bert", transform=None, pre_transform=None):
        self.root = root
        self.voc = vocabulary
        self.df = df.copy()
        self.use_cuda = config.use_cuda
        self.config = config
        
        assert embedd_mode in ["random", "bert"] # I only provide a text transformed dataset for random and bert embeddings
        if embedd_mode == "random":
            # Note: any modification for the random embeddings may be required.
            self.df[config.nodes] = [self.words_to_indices(row) for row in self.df[config.nodes]] # transform text to index first to make the model faster
        elif embedd_mode == "bert":
            bertembedding = BertEmbedding()
            pipe = Pipe([bertembedding.get_embeddings, torch.tensor])
            self.build_embeddings = pipe.call_pipeline
        self.embedd_mode = embedd_mode
            
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        pass

    def len(self):
        return self.df.shape[0]

    def get(self, idx):
        return self.build_data(idx)
    
    def build_data(self, idx):
        """
        This function is to build Pytorch Geometric Data by a given sample index.

        @Arguments:
            idx (int): an index of the sample
        @Return:
            data (Pytorch Geometric Data)
        """
        tt = torch.cuda if self.use_cuda else torch
        if self.embedd_mode == "random":
            x = torch.tensor(self.df[self.config.nodes][idx]) 
        elif self.embedd_mode == "bert":
            x = self.build_embeddings(self.df[self.config.nodes][idx]).float().to(self.config.device) # build embedding matrix from "texts"
        edge_index = tt.LongTensor(self.df[self.config.edge_index][idx]) # "config.EDGE_INDEX == edgeIndex"
        y = tt.FloatTensor([self.df[self.config.target][idx]]) ## config.TARGET == "y" or "overall"
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def words_to_indices(self, words):
        """
        This function turns words to indices from Vocabulary list and returns a tensor of indices.
        
        @Arguments:
            words: a list of words
        @Return:
            indices: a list of indices
        """
        indices = [self.voc.get_index(w) for w in words]
        return indices


def get_datasets(config, root, vocabulary, embedd_mode, train_df, val_df, test_df):
    """
    This function is to get pytorch dataset and only used for the GraphDataset class.

    @Arguments:
        config (dictionary): the config setting
        root (string): a root path is required but may not be used
        vocabulary (Vocabulary): the custom Vocabulary class object
        embedd_mode (string): should be set to "bert" or "random" only (modify it yourself if you plan to use other embedding methods)
        train_df (pd.DataFrame): the training set/dataframe
        val_df (pd.DataFrame): the validation set/dataframe
        test_df (pd.DataFrame): the testing set/dataframe
    """
    datasets = []
    for data in [train_df, val_df, test_df]:
        datasets.append(GraphDataset(config, root, data, vocabulary, embedd_mode))
    return datasets


"""
2. Use this one if you cannot load the whole dataset at one time.
"""
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
class GraphDataset_A(Dataset):
    def __init__(self, df, root, use_cuda, embeddings=None, transform=None, pre_transform=None):
        self.root = root
        self.df = df
        self.use_cuda = use_cuda
        self.embeddings = embeddings
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data_{}.pt" for i in range(self.df.shape[0])]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        for i in range(self.df.shape[0]):
            data = self.build_data(i)
            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
    def build_data(self, idx):
        tt = torch.cuda if self.use_cuda else torch
        edge_index = tt.LongTensor(self.df["edgeIndex2"][idx])
        x = tt.FloatTensor(self.embeddings.get_embeddings(self.df["uniqueTokens"][idx]))
        y = tt.FloatTensor([self.df["overall"][idx]])
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    
"""
3. InMemoryDataset is used for small datasets only.
"""
class GraphDataset_B(InMemoryDataset):
    def __init__(self, root, input_data_path, df, set_seed=123, use_cuda=True, train_val_percent=[0.8, 0.1], task="train", transform=None, pre_transform=None, pre_filter=None):
        self.input_data_path = input_data_path
        self.df = df
        self.set_seed = set_seed
        self.use_cuda = use_cuda
        self.train_val_percent = train_val_percent # list: the percentage is used to split the dataset into train, validation, and test sets.
        self.task = task # ["train", "validation", "test"] -> sepecify which set you want to obtain
        
        assert self.task in ["train", "validation", "test"] # must sepecify the correct name of the task
        assert sum(self.train_val_percent) < 1 # percentage of the test set == 1-sum(self.train_val_percent)
        
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        if self.task == "train":
            return ["train.pt"]
        elif self.task == "validation":
            return ["val.pt"]
        elif self.task == "test":
            return ["test.pt"]
    
    def download(self):
        ''' down data to your current working directory '''
        pass
    
    def process(self):
#         df = pd.read_json(self.input_data_path)
        data_list = []
        tt = torch.cuda if self.use_cuda else torch
        
        self.df = shuffle(df, random_state=self.set_seed)
        train_split_index = int(len(self.df)*self.train_val_percent[0])
        val_split_index = int(len(self.df)*sum(self.train_val_percent))
        
        if self.task == "train":
            df_subset = self.df[:train_split_index].reset_index(drop=True)
        elif self.task == "validation":
            df_subset = self.df[train_split_index:val_split_index].reset_index(drop=True)
        elif self.task == "test":
            df_subset = self.df[val_split_index:].reset_index(drop=True)

        bertembeddings = BertEmbedding()
        pool = mp.Pool(20)
        data_list = pool.map(self.build_data, tqdm(range(df_subset.shape[0])))
        pool.close()
        pool.join()
        
        # save the data
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def build_data(self, idx):
        edge_index = tt.LongTensor(df_subset["edgeIndex2"][idx])
        x = tt.LongTensor(bertembeddings.get_embeddings(df_subset["uniqueTokens"][idx]))
        y = tt.FloatTensor([df_subset["overall"][idx]])
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
