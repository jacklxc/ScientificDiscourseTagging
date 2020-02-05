# Scientific Discourse Tagging

Scientific discourse tagger implementation for paper "Scientific Discourse Tagging and Applications".

This is a discourse tagger that tags each clause in a given paragraph from a biomedical paper with 8 types of discourse types, such as "fact", "method", "result", "implication" etc. The paper will be available as soon as it's published.

If you have any question, please contact xiangcil@usc.edu.

## Requirements
* Python 3
* Tensorflow (tested with v1.12, either gpu or non-gpu version is fine)
* Keras (tested with v2.2.4, either gpu or non-gpu version is fine)
* Scikit-learn
* Keras-bert
* Pre-trained embedding
  * [BioBERT] (https://github.com/dmis-lab/biobert)
  * [SciBERT] (https://github.com/allenai/scibert)

## Input Format
Our discourse tagger expects inputs to lists of clauses or sentences, with paragraph boundaries identified, i.e., each line in the input file needs to be a clause or sentence and paragraphs should be separated by blank lines.

If you are training, the file additionally needs labels at the clause level, which can be specified on each line, after the clause, separated by a tab. 

## Intended Usage
As explained in the paper, the model is intended for tagging discourse elements in biomedical research papers, and we use the seven label taxonomy described in [De Waard and Pander Maat (2012)](http://www.sciencedirect.com/science/article/pii/S1475158512000471) and additional "none" label for SciDT dataset and [PubMed-RCT](https://github.com/Franck-Dernoncourt/pubmed-rct). For the detailed usage, check our paper.

## Steps to use
### Preparing word embeddings

#### BioBERT embedding
* Follow the instruction of [SciBERT] (https://github.com/allenai/scibert), download the pretrained weights. Remeber to correct the file names to match the hard-coded name in BERT code.

## Training
### SciBERT embedding
```
python -u discourse_tagger_generator_bert.py --repfile REPFILE --train_file TRAINFILE --validation_file DEVFILE  --use_attention --att_context LSTM_clause --bidirectional --crf --save --maxseqlen 40 --maxclauselen 60
```
where `REPFILE` is the BERT embedding path. `--use_attention` is recommended. `--att_context` is the type of attention. `--bidirectional` means use bidirectional LSTM for sequence tagger. `--crf` means use CRF as the last layer. `--save` to save the trained model. Check out the help messages for `discourse_tagger_generator_bert.py` for more options.

### Trained model
After you train successfully, three new files appear in the directory, with file names starting `model_`.

## Testing
You can specify test files while training itself using `--test_files` arguments. Alternatively, you can do it after training is done. In the latter case, `discourse_tagger` assumes the trained model files described above are present in the directory.
```
python -u discourse_tagger_generator_bert.py --repfile REPFILE --test_file TESTFILE1 [TESTFILE2 ..] --use_attention --att_context LSTM_clause --bidirectional --crf --maxseqlen 40 --maxclauselen 60
```
Make sure you use the same options for attention, context and bidirectional as you used for training.
