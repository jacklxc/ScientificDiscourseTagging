import warnings
import sys
import codecs
import numpy as np
import argparse
import json
import pickle

from rep_reader import RepReader
from util import read_passages 
from util import evaluate, make_folds, clean_words, test_f1, to_BIO, from_BIO, from_BIO_ind

import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)
from keras.activations import softmax
from keras.regularizers import l2
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.optimizers import Adam, RMSprop, SGD
from crf import CRF
from attention import TensorAttention
from custom_layers import HigherOrderTimeDistributedDense

def reset_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

class PassageTagger(object):
    def __init__(self, word_rep_file=None, pickled_rep_reader=None):
        if pickled_rep_reader:
            self.rep_reader = pickled_rep_reader
        elif word_rep_file:
            self.rep_reader = RepReader(word_rep_file)
        self.input_size = self.rep_reader.rep_shape[0]
        self.tagger = None
    
    def make_data(self, trainfilename, use_attention, maxseqlen=None, maxclauselen=None, label_ind=None, train=False):
        str_seqs, label_seqs = read_passages(trainfilename, is_labeled=train)
        print("Filtering data")
        str_seqs = clean_words(str_seqs)
        label_seqs = to_BIO(label_seqs)
        if not label_ind:
            self.label_ind = {"none": 0}
        else:
            self.label_ind = label_ind
        seq_lengths = [len(seq) for seq in str_seqs]
        if not maxseqlen:
            maxseqlen = max(seq_lengths)
        if not maxclauselen:
            if use_attention:
                clauselens = []
                for str_seq in str_seqs:
                    clauselens.extend([len(clause.split()) for clause in str_seq])
                    
                maxclauselen = np.round(np.mean(clauselens) + 3 * np.std(clauselens)).astype(int)
        X = []
        Y = []
        Y_inds = []
        init_word_rep_len = len(self.rep_reader.word_rep) # Vocab size
        all_word_types = set([])
        for str_seq, label_seq in zip(str_seqs, label_seqs):
            for label in label_seq:
                if label not in self.label_ind:
                    # Add new labels with values 0,1,2,....
                    self.label_ind[label] = len(self.label_ind)
            if use_attention:
                x = np.zeros((maxseqlen, maxclauselen, self.input_size))
            else:
                x = np.zeros((maxseqlen, self.input_size))
            y_ind = np.zeros(maxseqlen)
            seq_len = len(str_seq)
            # The following conditional is true only when we've already trained, and one of the sequences in the test set is longer than the longest sequence in training.
            if seq_len > maxseqlen:
                str_seq = str_seq[:maxseqlen]
                seq_len = maxseqlen
            if train:
                for i, (clause, label) in enumerate(zip(str_seq, label_seq)):
                    clause_rep = self.rep_reader.get_clause_rep(clause.lower()) # Makes embedding non-trainable from the beginning.
                    for word in clause.split():
                        all_word_types.add(word) # Vocab
                    if use_attention:
                        if len(clause_rep) > maxclauselen:
                            clause_rep = clause_rep[:maxclauselen]
                        x[-seq_len+i][-len(clause_rep):] = clause_rep
                    else:
                        x[-seq_len+i] = np.max(clause_rep, axis=0)
                    y_ind[-seq_len+i] = self.label_ind[label]
                X.append(x)
                Y_inds.append(y_ind)
            else:
                for i, clause in enumerate(str_seq):
                    clause_rep = self.rep_reader.get_clause_rep(clause.lower())
                    for word in clause.split():
                        all_word_types.add(word)
                    if use_attention:
                        if len(clause_rep) > maxclauselen:
                            clause_rep = clause_rep[:maxclauselen]
                        x[-seq_len+i][-len(clause_rep):] = clause_rep
                    else:
                        x[-seq_len+i] = np.mean(clause_rep, axis=0)
                X.append(x)
        # Once there is OOV, new word vector is added to word_rep
        final_word_rep_len = len(self.rep_reader.word_rep)
        oov_ratio = float(final_word_rep_len - init_word_rep_len)/len(all_word_types)
        print("OOV ratio:",oov_ratio)
        for y_ind in Y_inds:
            y = np.zeros((maxseqlen, len(self.label_ind)))
            for i, y_ind_i in enumerate(y_ind):
                y[i][y_ind_i.astype(int)] = 1
            Y.append(y)
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        return seq_lengths, np.asarray(X), np.asarray(Y) # One-hot representation of labels

    def predict(self, X, test_seq_lengths=None, tagger=None, batch_size=None):
        if not tagger:
            tagger = self.tagger
        if test_seq_lengths is None:
            # Determining actual lengths sans padding
            x_lens = []
            for x in X:
                x_len = 0
                for i, xi in enumerate(x):
                    if xi.sum() != 0:
                        x_len = len(x) - i
                        break
                x_lens.append(x_len)
        else:
                x_lens = test_seq_lengths
        pred_probs = tagger.predict(X, batch_size=batch_size)
        pred_inds = np.argmax(pred_probs, axis=2)
        pred_label_seqs = []
        for pred_ind, x_len in zip(pred_inds, x_lens):
            pred_label_seq = [self.rev_label_ind[pred] for pred in pred_ind][-x_len:]
            # If the following number is positive, it means we ignored some clauses in the test passage to make it the same length as the ones we trained on.
            num_ignored_clauses = max(0, x_len - len(pred_label_seq))
            # Make labels for those if needed.
            if num_ignored_clauses > 0:
                warnings.warn("Test sequence too long. Ignoring %d clauses at the beginning and labeling them none." % num_ignored_clauses)
                ignored_clause_labels = ["none"] * num_ignored_clauses
                pred_label_seq = ignored_clause_labels + pred_label_seq
            pred_label_seqs.append(pred_label_seq)
        return pred_probs, pred_label_seqs, x_lens

    def fit_model(self, X, Y, use_attention, att_context, lstm, bidirectional, crf, embedding_dropout=0, high_dense_dropout=0, attention_dropout=0, lstm_dropout = 0, word_proj_dim=50, lstm_dim=50, lr=1e-3,epoch=100, batch_size=10, hard_k=0, reg=0, validation_split = 0.1, att_proj_dim = None, rec_hid_dim = None):
        early_stopping = EarlyStopping(patience = 5)
        num_classes = len(self.label_ind)
        tagger = Sequential()
        #tagger.add(Masking(mask_value=0.0))
        tagger.add(Dropout(embedding_dropout)) 
        if use_attention:
            _, input_len, timesteps, input_dim = X.shape
            tagger.add(HigherOrderTimeDistributedDense(input_dim=input_dim, output_dim=word_proj_dim, reg=reg))
            att_input_shape = (input_len, timesteps, word_proj_dim)
            tagger.add(Dropout(high_dense_dropout)) 
            tagger.add(TensorAttention(att_input_shape, context=att_context, hard_k=hard_k, proj_dim = att_proj_dim, rec_hid_dim = rec_hid_dim))
            tagger.add(Dropout(attention_dropout))
        else:
            _, input_len, input_dim = X.shape
            tagger.add(TimeDistributed(Dense(input_dim=input_dim, units=word_proj_dim)))
        
        if bidirectional:
            tagger.add(Bidirectional(LSTM(input_shape=(input_len,word_proj_dim), units=lstm_dim, 
                                          return_sequences=True,kernel_regularizer=l2(reg),
                                          recurrent_regularizer=l2(reg), 
                                          bias_regularizer=l2(reg))))
            tagger.add(Dropout(lstm_dropout)) 
        elif lstm:
            tagger.add(LSTM(input_shape=(input_len,word_proj_dim), units=lstm_dim, return_sequences=True,
                            kernel_regularizer=l2(reg),
                            recurrent_regularizer=l2(reg), 
                            bias_regularizer=l2(reg)))
            tagger.add(Dropout(lstm_dropout))

        if crf:
            Crf = CRF(num_classes,learn_mode="join")
            tagger.add(Crf)            
        else:
            tagger.add(TimeDistributed(Dense(num_classes, activation='softmax')))
                    
        def step_decay(current_epoch):
            initial_lrate = lr
            drop = 0.5
            epochs_drop = epoch/2
            lrate = initial_lrate * np.power(drop,  
                   np.floor((1+current_epoch)/epochs_drop))
            return lrate
        
        lr_fractions = [1,0.1]
        decay = 0
        for lr_fraction in lr_fractions:
            adam = Adam(lr=lr*lr_fraction, decay = decay)
            if crf:
                #rmsprop = RMSprop(lr=lr,decay = decay)
                tagger.compile(optimizer=adam, loss=Crf.loss_function, metrics=[Crf.accuracy])
            else:
                tagger.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            tagger.fit(X, Y, validation_split=validation_split, epochs=epoch, callbacks=[early_stopping], batch_size=batch_size,verbose=2)

        tagger.summary()
        return tagger

    def train(self, X, Y, use_attention, att_context, lstm, bidirectional, cv=True, folds=5, crf=False,save=True,embedding_dropout=0, high_dense_dropout=0, attention_dropout=0, lstm_dropout = 0, word_proj_dim=50,lr=1e-3,epoch=100, batch_size=10, hard_k=0, att_proj_dim = None, rec_hid_dim = None, lstm_dim = 50, validation_split = 0.1):
        f_mean, f_std, original_f_mean, original_f_std = 0,0,0,0
        if cv:
            cv_folds = make_folds(X, Y, folds)
            accuracies = []
            fscores = []
            original_accuracies = []
            original_fscores = []
            if save:
                cv_targets = []
                cv_preds = []

            for fold_num, ((train_fold_X, train_fold_Y), (test_fold_X, test_fold_Y)) in enumerate(cv_folds):
                self.tagger = self.fit_model(train_fold_X, train_fold_Y, use_attention, att_context, lstm, bidirectional, crf,embedding_dropout=embedding_dropout, high_dense_dropout=high_dense_dropout, attention_dropout=attention_dropout, lstm_dropout = lstm_dropout, word_proj_dim=word_proj_dim,lr=lr,epoch=epoch, batch_size=batch_size, hard_k=hard_k, att_proj_dim = att_proj_dim, rec_hid_dim = rec_hid_dim, lstm_dim=lstm_dim, validation_split=validation_split)
                pred_probs, pred_label_seqs, x_lens = self.predict(test_fold_X, tagger=self.tagger, batch_size=batch_size)
                # Free the tagger from memory.
                del self.tagger
                K.clear_session()
                self.tagger=None
                
                pred_inds = np.argmax(pred_probs, axis=2)
                flattened_preds = []
                flattened_targets = []
                for x_len, pred_ind, test_target in zip(x_lens, pred_inds, test_fold_Y):
                    flattened_preds.extend(pred_ind[-x_len:])
                    flattened_targets.extend([list(tt).index(1) for tt in test_target[-x_len:]])
                assert len(flattened_preds) == len(flattened_targets)
                original_preds, original_targets = from_BIO_ind(flattened_preds, flattened_targets, self.label_ind)
                assert len(original_preds) == len(original_targets)
                if save:
                    cv_preds.append(original_preds)
                    cv_targets.append(original_targets)
                accuracy, weighted_fscore, all_fscores = evaluate(flattened_targets, flattened_preds)
                original_accuracy, original_weighted_fscore, original_all_fscores = evaluate(original_targets, original_preds)

                print("Finished fold %d. Accuracy: %f, Weighted F-score: %f"%(fold_num, accuracy, weighted_fscore))
                print("Finished fold %d. Original accuracy: %f, Original weighted F-score: %f"%(fold_num, original_accuracy, original_weighted_fscore))

                print("Individual f-scores:")
                for cat in all_fscores:
                    print("%s: %f"%(self.rev_label_ind[cat], all_fscores[cat]))
                accuracies.append(accuracy)
                fscores.append(weighted_fscore)
                original_accuracies.append(original_accuracy)
                original_fscores.append(original_weighted_fscore)
            accuracies = np.asarray(accuracies)
            fscores = np.asarray(fscores)
            original_accuracies = np.asarray(original_accuracies)
            original_fscores = np.asarray(original_fscores)
            print("Accuracies:", accuracies)
            print("Average: %0.4f (+/- %0.4f)"%(accuracies.mean(), accuracies.std() * 2))
            print("Fscores:", fscores)
            f_mean, f_std = fscores.mean(), fscores.std()
            print("Average: %0.4f (+/- %0.4f)"%(f_mean, f_std * 2))
            print("Original accuracies:", original_accuracies)
            print("Average: %0.4f (+/- %0.4f)"%(original_accuracies.mean(), original_accuracies.std() * 2))
            original_f_mean, original_f_std = original_fscores.mean(), original_fscores.std()
            print("Original Fscores:", original_fscores)
            print("Average: %0.4f (+/- %0.4f)"%(original_f_mean, original_f_std * 2))
            
        self.tagger = self.fit_model(X, Y, use_attention, att_context, lstm, bidirectional, crf,embedding_dropout=embedding_dropout, high_dense_dropout=high_dense_dropout, attention_dropout=attention_dropout, lstm_dropout = lstm_dropout, word_proj_dim=word_proj_dim, lr=lr,epoch=epoch, batch_size=batch_size, hard_k=hard_k,att_proj_dim = att_proj_dim, rec_hid_dim = rec_hid_dim, lstm_dim=lstm_dim, validation_split=validation_split)
        if save:
            model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(use_attention), att_context, str(lstm), str(bid), str(crf))
            model_config_file = open("model_%s_config.json"%model_ext, "w")
            model_weights_file_name = "model_%s_weights"%model_ext
            model_label_ind = "model_%s_label_ind.json"%model_ext
            model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
            print(self.tagger.to_json(), file=model_config_file)
            self.tagger.save_weights(model_weights_file_name, overwrite=True)
            json.dump(self.label_ind, open(model_label_ind, "w"))
            pickle.dump(self.rep_reader, open(model_rep_reader, "wb"))
            if cv:
                cv_targets_filename = "model_%s_cv_targets.pkl"%model_ext
                pickle.dump(cv_targets, open(cv_targets_filename, "wb"))
                cv_preds_filename = "model_%s_cv_preds.pkl"%model_ext
                pickle.dump(cv_preds, open(cv_preds_filename, "wb"))
        
        return f_mean, f_std, original_f_mean, original_f_std

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run LSTM discourse tagger")
    argparser.add_argument('--repfile', type=str, help="Word embedding file")
    argparser.add_argument('--train_file', type=str, help="Training file. One clause<tab>label per line and passages separated by blank lines.")
    argparser.add_argument('--cv', help="Do cross validation", action='store_true')
    argparser.add_argument('--test_files', metavar="TESTFILE", type=str, nargs='+', help="Test file name(s), separated by space. One clause per line and passages separated by blank lines.")
    argparser.add_argument('--use_attention', help="Use attention over words? Or else will average their representations", action='store_true')
    argparser.add_argument('--att_context', type=str, help="Context to look at for determining attention (word/clause)")
    argparser.set_defaults(att_context='word')
    argparser.add_argument('--lstm', help="Sentence level LSTM", action='store_true')
    argparser.add_argument('--bidirectional', help="Bidirectional LSTM", action='store_true')
    argparser.add_argument('--crf', help="Conditional Random Field", action='store_true')
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
    argparser.set_defaults(word_proj_dim=225)
    argparser.add_argument('--lstm_dim', help="Discourse level LSTM dimension")
    argparser.set_defaults(lstm_dim=200)
    argparser.add_argument('--att_proj_dim', help="Attention projection dimension")
    argparser.set_defaults(att_proj_dim=110)
    argparser.add_argument('--rec_hid_dim', help="Attention RNN hidden dimension")
    argparser.set_defaults(rec_hid_dim=60)
    argparser.add_argument('--epoch', help="Training epoch")
    argparser.set_defaults(epoch=100)
    argparser.add_argument('--validation_split', help="validation_split")
    argparser.set_defaults(validation_split=0.1)
    argparser.add_argument('--save', help="Whether save the model or not",action='store_true')
    argparser.add_argument('--maxseqlen', help="max number of clauses per paragraph")
    argparser.set_defaults(maxseqlen=40)
    argparser.add_argument('--maxclauselen', help="max number of words per clause")
    argparser.set_defaults(maxclauselen=60)
    argparser.add_argument('--outpath', help="path of output labels")
    argparser.set_defaults(outpath="./")
    argparser.add_argument('--batch_size', help="batch size")
    argparser.set_defaults(batch_size=10)
    
    args = argparser.parse_args()
    reset_random_seed(12345) # Good for word attention
    repfile = args.repfile
    if args.train_file:
        trainfile = args.train_file
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.test_files:
        testfiles = args.test_files
        test = True
    else:
        test = False

    if not train and not test:
        raise(RuntimeError, "Please specify a train file or test files.")
    use_attention = args.use_attention
    att_context = args.att_context
    lstm = args.lstm
    bid = args.bidirectional
    crf = args.crf
    lr = float(args.lr)
    hard_k = int(args.hard_k)
    embedding_dropout = float(args.embedding_dropout)
    high_dense_dropout = float(args.high_dense_dropout)
    attention_dropout = float(args.attention_dropout)
    lstm_dropout = float(args.lstm_dropout)
    word_proj_dim = int(args.word_proj_dim)
    lstm_dim = int(args.lstm_dim)
    att_proj_dim = int(args.att_proj_dim)
    rec_hid_dim = int(args.rec_hid_dim)
    epoch = int(args.epoch)
    maxseqlen = int(args.maxseqlen)
    maxclauselen = int(args.maxclauselen)
    batch_size=int(args.batch_size)
    if maxseqlen <= 0:
        maxseqlen = None
    if maxclauselen <= 0:
        maxclauselen = None
    
    save = args.save
    validation_split = float(args.validation_split)
    model_name = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(use_attention), att_context, str(lstm), str(bid),str(crf))
    print(model_name)
    f_mean, f_std, original_f_mean, original_f_std = 0,0,0,0
    if train:
        # First returned value is sequence lengths (without padding)
        nnt = PassageTagger(word_rep_file=repfile)
        if repfile:
            print("Using embedding weight to find embeddings.")
            _, X, Y = nnt.make_data(trainfile, use_attention, maxseqlen=maxseqlen, maxclauselen=maxclauselen, train=True)
        else:
            assert(0)
        f_mean, f_std, original_f_mean, original_f_std = nnt.train(X, Y, use_attention, att_context, lstm, bid, cv=args.cv,  crf=crf,save=save,embedding_dropout=embedding_dropout, high_dense_dropout=high_dense_dropout, attention_dropout=attention_dropout, lstm_dropout = lstm_dropout, word_proj_dim=word_proj_dim,lr=lr,epoch=epoch, hard_k=hard_k, lstm_dim = lstm_dim, rec_hid_dim=rec_hid_dim, att_proj_dim=att_proj_dim, validation_split=validation_split, batch_size=batch_size)
    if test:
        if train:
            label_ind = nnt.label_ind
        else:
            # Load the model from file
            model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(use_attention), att_context, str(lstm), str(bid), str(crf))
            model_config_file = open("model_%s_config.json"%model_ext, "r")
            model_weights_file_name = "model_%s_weights"%model_ext
            model_label_ind = "model_%s_label_ind.json"%model_ext
            model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
            rep_reader = pickle.load(open(model_rep_reader, "rb"))
            print("Loaded pickled rep reader")
            nnt = PassageTagger(pickled_rep_reader=rep_reader)
            nnt.tagger = model_from_json(model_config_file.read(), custom_objects={"TensorAttention":TensorAttention, "HigherOrderTimeDistributedDense":HigherOrderTimeDistributedDense,"CRF":CRF})
            print("Loaded model:")
            print(nnt.tagger.summary())
            nnt.tagger.load_weights(model_weights_file_name)
            print("Loaded weights")
            label_ind_json = json.load(open(model_label_ind))
            label_ind = {k: int(label_ind_json[k]) for k in label_ind_json}
            print("Loaded label index:", label_ind)
        if not use_attention:
            maxseqlen = nnt.tagger.inputs[0].shape[1]
            maxclauselen = None
        else:
            for l in nnt.tagger.layers:
                if ("TensorAttention" in l.name) or ("tensor_attention" in l.name):
                    maxseqlen, maxclauselen = l.td1, l.td2
                    break

        for test_file in testfiles:
            print("Predicting on file %s"%(test_file))
            test_out_file_name = "predictions/"+test_file.split("/")[-1].replace(".txt", "")+model_name+".out"
            #test_out_file_name = args.outpath+test_file.split("/")[-1].replace(".txt", "")+"_GloVe"+".out"
            outfile = open(test_out_file_name, "w")
            print("maxseqlen", maxseqlen)
      
            test_seq_lengths, X_test, Y_test = nnt.make_data(test_file, use_attention, maxseqlen=maxseqlen, maxclauselen=maxclauselen, label_ind=label_ind, train=False)
            pred_probs, pred_label_seqs, _ = nnt.predict(X_test, test_seq_lengths, batch_size=batch_size)
                
            pred_label_seqs = from_BIO(pred_label_seqs)
            for pred_label_seq in pred_label_seqs:
                for pred_label in pred_label_seq:
                    print(pred_label,file = outfile)
                print("",file = outfile)

