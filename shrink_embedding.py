import numpy as np
from util import read_passages_original

"""
A simple script for shrinking the size of necessary static embeddings, e.g. GloVe embedding.
When the deep learning model uses word embeddings, it first loads all word vectors from the file to the memory.
This step is very slow. So when you are using the same dataset over and over again (for training and testing), it's faster to 
shrink the size of the necessary embedding file by simply store all words appeared in the whole dataset (train + test).
"""

all_data_name = "input_data.txt"
path = "/path/to/embedding/"
embedding_file = "embedding_file.txt"

str_seqs, label_seqs = read_passages_original(all_data_name, is_labeled=True)
all_para = [" ".join(seq_para) for seq_para in str_seqs]
all_str = " ".join(all_para)
vocab = set(all_str.split())
word_count = 0
wrote_count = 0
with open(path+"discourse_"+embedding_file, "w") as fwrite: 
    with open(path+embedding_file) as f:
        for line in f:
            word_count += 1 
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue
            if word in vocab:
                fwrite.write(line)
                wrote_count+=1
            if word_count % 10000 ==0:
                print(word_count)
