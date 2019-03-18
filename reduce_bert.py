import json
import numpy as np
import pickle
from sklearn.decomposition import PCA

PCA_name = "/path/to/PCA_object.pkl"
my_data = "/path/to/BERT_output.json"
output_name = "/path/to/output.pkl"

with open(PCA_name,"rb") as f:
    pca = pickle.load(f, encoding='latin1')
print("Covariance matrix shape: ",pca.components_.shape)
print("Explained variance ratio: ",np.sum(pca.explained_variance_ratio_))
bert_dim = pca.components_.shape[1]
whole_matrices = [] # should have a paragraph level
para_matrices = []
with open(my_data) as f:
    for line in f:
        json_sentence = json.loads(line)
        sentence_len = len(json_sentence["features"])
        if sentence_len>2: # More than [CLS] and [SEP]
            sentence_matrix = np.zeros((sentence_len, bert_dim))
            word_index = 0
            for i_word in range(sentence_len):
                sentence_matrix[word_index,:] = np.array(json_sentence["features"][i_word]["layers"][0]["values"])
                word_index += 1
            para_matrices.append(pca.transform(sentence_matrix))
        else:
            whole_matrices.append(para_matrices)
            para_matrices = []
    whole_matrices.append(para_matrices)

with open(output_name,"wb") as f:
    pickle.dump(whole_matrices,f)
print("Done!")