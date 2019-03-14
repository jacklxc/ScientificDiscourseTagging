import json
import numpy as np
import pickle
from sklearn.decomposition import PCA

"""
A simple script to convert BERT output (json file) to Pickle file for discourse tagger to read as input.
This is only used when NO PCA dimension reduction is conducted. Therefore the vectors in pkl. is still 768-dimensional.
Assume json file only contains single layer output.
"""

my_data = "path/to/BERT_activation/BERT.json"
output_name = "path/to/output.pkl"

bert_dim = 768
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
                sentence_matrix[word_index,:] = np.array(json_sentence["features"][i_word]["layers"][0]["values"]) # 0 assumes json file only contains single layer output.
                word_index += 1
            para_matrices.append(sentence_matrix)
        else:
            whole_matrices.append(para_matrices)
            para_matrices = []
    whole_matrices.append(para_matrices)

with open(output_name,"wb") as f:
    pickle.dump(whole_matrices,f)
print("Done!")
