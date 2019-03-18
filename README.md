# Discourse Tagging for Scientific Evidence Extraction

This is a discourse tagger that tags each clause in a given paragraph from a biomedical paper with 8 types of discourse types, such as "fact", "method", "result", "implication" etc. The paper will be available as soon as it's published.

If you have any question, please contact xiangcil@usc.edu.

## Requirements
* Tensorflow (tested with v1.12, either gpu or non-gpu version is fine)
* Keras (tested with v2.2.4, either gpu or non-gpu version is fine)
* Scikit-learn
* Pre-trained / Cached embedding
  * [BioGloVe] (https://zenodo.org/record/2530898#.XIs1ihNKgWp)
  * [BioBERT] (https://github.com/dmis-lab/biobert)

## Input Format
Our discourse tagger expects inputs to lists of clauses, with paragraph boundaries identified, i.e., each line in the input file needs to be a clause and and paragraphs should be separated by blank lines.

If you are training, the file additionally needs labels at the clause level, which can be specified on each line, after the clause, separated by a tab. 

## Intended Usage
As explained in the paper, the model is intended for tagging discourse elements in biomedical research papers, and we use the seven label taxonomy described in [De Waard and Pander Maat (2012)](http://www.sciencedirect.com/science/article/pii/S1475158512000471) and additional "none" label. A potential usage may be narrowing down sentences for scientific document summarization.

## Steps to use
### Preparing word embeddings
#### BioGloVe embedding
* Directly download pre-trained embedding.

#### BioBERT embedding
* Follow the instruction of [BioBERT] (https://github.com/dmis-lab/biobert), compute and save BioBERT embedding activations to json files. NOTE: Don't forget to exclude labels from the input txt file to BioBERT!
* If you want to directly use BioBERT embedding without PCA dimension reduction, edit the paths of the input and output files in `json2pkl.py` and run it to produce 768 dimensional BioBERT `pkl` file.
* If you want to do PCA dimension reduction to reduce 768 dimensional BioBERT embedding to 300 dimensions, download [`PCA.pkl`](https://zenodo.org/record/2530898#.XIs1ihNKgWp). Edit the paths in `reduce_bert.py` and run it to generate 300 dimensional reduced BioBERT activations in `pkl` format.

## Training
### BioGloVe embedding
```
python discourse_tagger.py --repfile REPFILE --train_file TRAINFILE --use_attention --att_context LSTM_clause --bidirectional --crf --save
```
where `REPFILE` is the embedding file. `--use_attention` is recommended. `--att_context` is the type of attention. `--bidirectional` means use bidirectional LSTM for sequence tagger. `--crf` means use CRF as the last layer. `--save` to save the trained model. Check out the help messages for `discourse_tagger.py` for more options.

### BioBERT embedding
Similarly, run
```
python discourse_tagger_bert.py --repfile REPFILE --train_file TRAINFILE --use_attention --att_context LSTM_clause --bidirectional --crf --save
```
to use BioBERT. `REPFILE` is the cached BioBERT pkl file. `TRAINFILE` is the txt file including text and label.

### Trained model
After you train successfully, three new files appear in the directory, with file names starting `model_`.

## Testing
You can specify test files while training itself using `--test_files` arguments. Alternatively, you can do it after training is done. In the latter case, `discourse_tagger` and `discourse_tagger_bert` assume the trained model files described above are present in the directory.
```
python discourse_tagger.py REPFILE --test_files TESTFILE1 [TESTFILE2 ..] --use_attention --att_context LSTM_clause --bidirectional --crf
```
Make sure you use the same options for attention, context and bidirectional as you used for training.
