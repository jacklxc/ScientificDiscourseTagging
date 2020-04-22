import os
import warnings
import sys
import codecs
import numpy as np
import argparse
import json
import pickle
import random
from glob import glob

from util import from_BIO, arg2param, read_passages

from crf import CRF
from attention import TensorAttention
from custom_layers import HigherOrderTimeDistributedDense
from discourse_tagger_generator_bert import PassageTagger, reset_random_seed
from keras.models import model_from_json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--repfile', type=str, help="Word embedding file")
    argparser.add_argument('--test_path', type=str, help="Test path name, one clause per line and passages separated by blank lines.")
    argparser.add_argument('--out_path', type=str, help="Output path name, one clause per line and passages separated by blank lines.")
    argparser.set_defaults(out_path="predictions/")
    argparser.add_argument('--use_attention', help="Use attention over words? Or else will average their representations", action='store_true')
    argparser.set_defaults(use_attention=True)
    argparser.add_argument('--att_context', type=str, help="Context to look at for determining attention (word/clause)")
    argparser.set_defaults(att_context='LSTM_clause')
    argparser.add_argument('--lstm', help="Sentence level LSTM", action='store_true')
    argparser.set_defaults(lstm=False)
    argparser.add_argument('--bidirectional', help="Bidirectional LSTM", action='store_true')
    argparser.set_defaults(bidirectional=True)
    argparser.add_argument('--crf', help="Conditional Random Field", action='store_true')
    argparser.set_defaults(crf=True)
    argparser.add_argument('--hard_k', help="Hard attention's choose top k")
    argparser.set_defaults(hard_k=0)
    argparser.add_argument('--lr', help="Learning rate")
    argparser.set_defaults(lr=1e-3)
    argparser.add_argument('--embedding_dropout', help="embedding_dropout rate")
    argparser.add_argument('--high_dense_dropout', help="high_dense_dropout rate")
    argparser.add_argument('--attention_dropout', help="attention_dropout rate")
    argparser.add_argument('--lstm_dropout', help="lstm_dropout rate")
    argparser.set_defaults(embedding_dropout=0.4)
    argparser.set_defaults(high_dense_dropout=0.4)
    argparser.set_defaults(attention_dropout=0.6)
    argparser.set_defaults(lstm_dropout=0.5)
    argparser.add_argument('--word_proj_dim', help="word_projection_dimension")
    argparser.set_defaults(word_proj_dim=300)
    argparser.add_argument('--lstm_dim', help="Discourse level LSTM dimension")
    argparser.set_defaults(lstm_dim=350)
    argparser.add_argument('--att_proj_dim', help="Attention projection dimension")
    argparser.set_defaults(att_proj_dim=200)
    argparser.add_argument('--rec_hid_dim', help="Attention RNN hidden dimension")
    argparser.set_defaults(rec_hid_dim=75)
    argparser.add_argument('--epoch', help="Training epoch")
    argparser.set_defaults(epoch=20)
    argparser.add_argument('--validation_split', help="validation_split")
    argparser.set_defaults(validation_split=0.1)
    argparser.add_argument('--maxseqlen', help="max number of clauses per paragraph")
    argparser.set_defaults(maxseqlen=40) ######## 40
    argparser.add_argument('--maxclauselen', help="max number of words per clause")
    argparser.set_defaults(maxclauselen=60) ########## 60
    argparser.add_argument('--outpath', help="path of output labels")
    argparser.set_defaults(outpath="./")
    argparser.add_argument('--batch_size', help="batch size")
    argparser.set_defaults(batch_size=100)
    
    args = argparser.parse_args()
    params = arg2param(args)
    reset_random_seed(12345) # Good for word attention

    params["train"] = False
    if args.test_path:
        params["test"] = True
    else:
        params["test"] = False

    if not params["train"] and not params["test"]:
        raise(RuntimeError, "Please specify files to predict.")

    if params["maxseqlen"] <= 0:
        params["maxseqlen"] = None
    if params["maxclauselen"] <= 0:
        params["maxclauselen"] = None
    
    model_name = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(params["use_attention"]), params["att_context"], str(params["lstm"]), str(params["bidirectional"]),str(params["crf"]))
    print(model_name)

    if params["test"]:

        # Load the model from file
        model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(params["use_attention"]), params["att_context"], str(params["lstm"]), str(params["bidirectional"]), str(params["crf"]))
        model_config_file = open("model_%s_config.json"%model_ext, "r")
        model_weights_file_name = "model_%s_weights"%model_ext
        model_label_ind = "model_%s_label_ind.json"%model_ext
        nnt = PassageTagger(params)
        nnt.tagger = model_from_json(model_config_file.read(), custom_objects={"TensorAttention":TensorAttention, "HigherOrderTimeDistributedDense":HigherOrderTimeDistributedDense,"CRF":CRF})
        print("Loaded model:")
        print(nnt.tagger.summary())
        nnt.tagger.load_weights(model_weights_file_name)
        print("Loaded weights")
        label_ind_json = json.load(open(model_label_ind))
        label_ind = {k: int(label_ind_json[k]) for k in label_ind_json}
        print("Loaded label index:", label_ind)
        if not params["use_attention"]:
            params["maxseqlen"] = nnt.tagger.inputs[0].shape[1]
            params["maxclauselen"] = None
        else:
            for l in nnt.tagger.layers:
                if ("TensorAttention" in l.name) or ("tensor_attention" in l.name):
                    params["maxseqlen"], params["maxclauselen"] = l.td1, l.td2
                    break

        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)

        paper_paths = glob(os.path.join(params["test_path"],"*"))
        random.shuffle(paper_paths)

        for test_file in paper_paths:
            test_out_file_name = os.path.join(params["out_path"], test_file.split("/")[-1].replace(".txt", "")+".tsv")

            if not os.path.exists(test_out_file_name):
                print("Predicting on file %s"%(test_file))
                raw_seqs, _ = read_passages(test_file, is_labeled=False)
                if len(raw_seqs)==0:
                    print("Empty file", test_file)
                    continue
                outfile = open(test_out_file_name, "w")
                
                test_seq_lengths, test_generator = nnt.make_data(test_file, label_ind=label_ind, train=False)
                pred_probs, pred_label_seqs, _ = nnt.predict(test_generator, test_seq_lengths)

                pred_label_seqs = from_BIO(pred_label_seqs)
                for raw_seq, pred_label_seq in zip(raw_seqs, pred_label_seqs):
                    for clause, pred_label in zip(raw_seq, pred_label_seq):
                        print(clause + "\t" + pred_label,file = outfile)
                    print("",file = outfile)
                outfile.close()
            else:
                print("%s already exists!"%(test_file))