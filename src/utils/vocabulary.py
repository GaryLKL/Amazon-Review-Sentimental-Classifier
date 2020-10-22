# refer: https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html
class Vocabulary:
    def __init__(self, PAD_token=0):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1
        self.pad = PAD_token
    
    def add_word(self, word):
        if word not in self.word2index:
            # words do not exist
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # words exist
            self.word2count[word] += 1
    
    def get_index(self, word):
        if self.word2index.get(word) is None:
            return self.pad
        else:
            return self.word2index[word]
    
    def get_word(self, index):
        assert index <= self.num_words
        return self.index2word[index]