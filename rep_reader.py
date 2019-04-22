import numpy as np

class RepReader(object):
    def __init__(self, EMBEDDING_DIR):
        print('Indexing word vectors.')
        self.MAX_NUM_WORDS = 3730000
        self.word_rep = {}
        word_count = 0
        with open(EMBEDDING_DIR) as f:
            for line in f:
                word_count += 1 
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
                self.word_rep[word] = coefs
                if word_count>self.MAX_NUM_WORDS:
                    break
            self.rep_shape = coefs.shape
        self.rep_min = min([x.min() for x in self.word_rep.values()])
        self.rep_max = min([x.max() for x in self.word_rep.values()])
        self.numpy_rng = np.random.RandomState(12345)
    
    def get_clause_rep(self, clause):
        reps = []
        for word in clause.split():
            if word not in self.word_rep:
                # Embed OOV by randomly assign a vector
                rep = self.numpy_rng.uniform(low = self.rep_min, high = self.rep_max, size = self.rep_shape)
                self.word_rep[word] = rep
            else:
                rep = self.word_rep[word]
            reps.append(rep)
        return np.asarray(reps)
