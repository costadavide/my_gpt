import re 


class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

        self.pat = re.compile(r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+')

    def pre_tokenize(self, text):

        return self.pat.findall(text)
        

    def train(self, corpus):
        pass

    def encode(self, text):
        pass

    def decode(self, token_ids):
        pass



        
                