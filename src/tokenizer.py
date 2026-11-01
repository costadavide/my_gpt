import re 
import heapq


class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        self.pat = re.compile(r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+')

    def pre_tokenize(self, text):
        return self.pat.findall(text)

    def get_pair_counts(self, corpus_in_bytes):
        pair_counts = {}

        for w in corpus_in_bytes:
            for i in range(1, len(w)):
                pair_counts[(w[i-1], w[i])] = pair_counts.get((w[i-1], w[i]), 0) + 1

        return pair_counts 

    def merge_pair(self, word, pair, new_id):
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word)-1 and (word[i], word[i+1]) == pair:
                new_word.append(new_id)
                i += 2 
            else: 
                new_word.append(word[i])
                i += 1 

        return new_word

    def train(self, corpus):
        if self.merges:
            raise RuntimeError("Tokenizer has already been trained")
        
        corpus = self.pre_tokenize(corpus)
        corpus_in_bytes = [list(word.encode("utf-8")) for word in corpus]
        
        while len(self.vocab) < self.vocab_size:
            count_freq_adj = self.get_pair_counts(corpus_in_bytes)

            if not count_freq_adj: 
                break 

            most_freq_pair = max(count_freq_adj, key=count_freq_adj.get)
            new_id = len(self.vocab)

            self.vocab[new_id] = self.vocab[most_freq_pair[0]]+self.vocab[most_freq_pair[1]]
            self.merges[most_freq_pair] = new_id

            for i, word in enumerate(corpus_in_bytes): 
                new_word = self.merge_pair(word, most_freq_pair, new_id)
                corpus_in_bytes[i] = new_word

    def encode(self, text):
        words = self.pre_tokenize(text)
        token_ids = []

        for word in words: 
            tokens = list(word.encode("utf-8"))
            n = len(tokens)

            if n < 2:
                token_ids.extend(tokens)
                continue

            while True: 
                pairs = (
                (tokens[i], tokens[i + 1])
                for i in range(len(tokens) - 1)
                if (tokens[i], tokens[i + 1]) in self.merges
                )

                pair = min(pairs, key=self.merges.get, default=None)
                if pair is None:
                    break

                new_id = self.merges[pair]
                tokens = self.merge_pair(tokens, pair, new_id)

            token_ids.extend(tokens)

        return token_ids 

    def decode(self, token_ids):
        word = b"".join(self.vocab[token] for token in token_ids)
        return word.decode("utf-8")




        