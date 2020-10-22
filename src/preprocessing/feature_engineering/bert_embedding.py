from transformers import BertTokenizer, BertModel

class BertEmbedding:
    def __init__(self, max_len=None):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.embedding_matrix = self.get_bert_embed_matrix()
        
    def get_bert_embed_matrix(self):
        bert = BertModel.from_pretrained('bert-base-uncased')
        bert_embeddings = list(bert.children())[0]
        bert_word_embeddings = list(bert_embeddings.children())[0]
        mat = bert_word_embeddings.weight.data.numpy()
        return mat

    def get_embeddings(self, row_data):
        input_ids = self.get_bert_index(row_data)
        embeddings = []
        for index in input_ids:
            embeddings.append(self.embedding_matrix[index])

        assert len(embeddings) == len(input_ids) and len(embeddings[0]) == self.embedding_matrix.shape[1]
        return embeddings
    
    def get_bert_index(self, row_data):
        '''
        @ param, row_data: a unique token list
        '''
        if len(row_data) == 0:
            return []
        
        if self.max_len is None:
            MAX_LEN = len(row_data)+2 # +2 is for adding cls and \cls
        else:
            MAX_LEN = self.max_len+2
            
        input_ids = self.tokenizer.encode(row_data,
                                          max_length=MAX_LEN,
                                          truncation=True,
                                          pad_to_max_length=True
                                         )
#         input_ids = input_ids[1:-1] # however, we do not take cls & \cls into consideration when building the embeddings
        return input_ids
    
# https://www.kaggle.com/mlwhiz/bilstm-pytorch-and-keras