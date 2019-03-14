import json
import numpy as np
import pickle
from sklearn.decomposition import PCA


"""
This script computes the PCA transformation matrix from the BERT activations of large unlabeled dataset.
Saves the whole sklearn PCA object as pickle file.
"""

large_corpus = '/path/to/large_BERT_output.json'
PCA_name = "/path/to/PCA_object.pkl"

n_components = 300
print("Reading JSON")
word_count = 0
print_threshold = 1
print_cycle = 100000
with open(large_corpus) as f:
    for line in f:
        json_sentence = json.loads(line)
        word_count += len(json_sentence["features"])
        if word_count > print_threshold*print_cycle:
            print(word_count)
            print_threshold += 1
bert_dim = len(json_sentence["features"][0]["layers"][0]["values"])

print("Word count: ",word_count)
print("Making large embedding matrix")

bert_words = np.zeros((word_count, bert_dim))

word_index = 0
with open(large_corpus) as f:
    for line in f:
        json_sentence = json.loads(line)
        for i_word in range(len(json_sentence["features"])):
            bert_words[word_index,:] = np.array(json_sentence["features"][i_word]["layers"][0]["values"])
            word_index += 1

print("Corpus embedding dimension: ",bert_words.shape)
        
pca = PCA(n_components=n_components)
pca.fit(bert_words)
print("Covariance matrix shape: ",pca.components_.shape)
print("Explained variance ratio: ",np.sum(pca.explained_variance_ratio_))
with open(PCA_name,"wb") as f:
    pickle.dump(pca,f)
