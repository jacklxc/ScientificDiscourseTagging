import numpy as np
from keras.utils import Sequence
import codecs

class DiscourseGenerator(Sequence):

    def __init__(self, rep_reader, str_seqs, label_seqs, label_ind, batch_size, use_attention, maxseqlen, maxclauselen, train, input_size):
        self.rep_reader = rep_reader
        self.str_seqs, self.label_seqs = str_seqs, label_seqs
        self.label_ind = label_ind
        self.batch_size = batch_size
        self.use_attention = use_attention
        self.maxseqlen = maxseqlen
        self.maxclauselen = maxclauselen
        self.train = train
        self.input_size = input_size

    def __len__(self):
        return int(np.ceil(len(self.str_seqs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_str_seqs = self.str_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_seqs = self.label_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X, batch_Y = self.make_data(batch_str_seqs, batch_label_seqs)
        return batch_X, batch_Y
    
    def make_data(self,str_seqs, label_seqs):
        if self.train:
            return self.make_data_train(str_seqs, label_seqs)
        else:
            return self.make_data_test(str_seqs)
    
    def make_data_train(self,str_seqs, label_seqs):
        X = []
        Y = []
        Y_inds = []
        for str_seq, label_seq in zip(str_seqs, label_seqs):
            if self.use_attention:
                x = np.zeros((self.maxseqlen, self.maxclauselen, self.input_size))
            else:
                x = np.zeros((self.maxseqlen, self.input_size))
            y_ind = np.zeros(self.maxseqlen)
            seq_len = len(str_seq)
            # The following conditional is true only when we've already trained, and one of the sequences in the test set is longer than the longest sequence in training.
            if seq_len > self.maxseqlen:
                str_seq = str_seq[:self.maxseqlen]
                seq_len = self.maxseqlen
            for i, (clause, label) in enumerate(zip(str_seq, label_seq)):
                clause_rep = self.rep_reader.get_clause_rep(clause.lower()) # Makes embedding non-trainable from the beginning.
                if self.use_attention:
                    if len(clause_rep) > self.maxclauselen:
                        clause_rep = clause_rep[:self.maxclauselen]
                    x[-seq_len+i][-len(clause_rep):] = clause_rep
                else:
                    #x[-seq_len+i] = np.max(clause_rep, axis=0)
                    x[-seq_len+i] = clause_rep[0,:]
                y_ind[-seq_len+i] = self.label_ind[label]
            X.append(x)
            Y_inds.append(y_ind)

        for y_ind in Y_inds:
            y = np.zeros((self.maxseqlen, len(self.label_ind)))
            for i, y_ind_i in enumerate(y_ind):
                y[i][y_ind_i.astype(int)] = 1
            Y.append(y)
        
        return np.asarray(X), np.asarray(Y) # One-hot representation of labels

    def make_data_test(self,str_seqs):
        X = []
        for str_seq in str_seqs:
            if self.use_attention:
                x = np.zeros((self.maxseqlen, self.maxclauselen, self.input_size))
            else:
                x = np.zeros((self.maxseqlen, self.input_size))
            seq_len = len(str_seq)
            # The following conditional is true only when we've already trained, and one of the sequences in the test set is longer than the longest sequence in training.
            if seq_len > self.maxseqlen:
                str_seq = str_seq[:self.maxseqlen]
                seq_len = self.maxseqlen
            for i, clause in enumerate(str_seq):
                clause_rep = self.rep_reader.get_clause_rep(clause.lower())
                if self.use_attention:
                    if len(clause_rep) > self.maxclauselen:
                        clause_rep = clause_rep[:self.maxclauselen]
                    x[-seq_len+i][-len(clause_rep):] = clause_rep
                else:
                    #x[-seq_len+i] = np.mean(clause_rep, axis=0)
                    x[-seq_len+i] = clause_rep[0,:]
            X.append(x)
        return np.asarray(X), np.asarray([]) # One-hot representation of labels

class BertDiscourseGenerator(Sequence):

    def __init__(self, bert, tokenizer, str_seqs, label_seqs, label_ind, batch_size, use_attention, maxseqlen, maxclauselen, train, input_size=768):
        
        self.bert = bert
        self.tokenizer = tokenizer
        
        self.str_seqs, self.label_seqs = str_seqs, label_seqs
        self.label_ind = label_ind
        self.batch_size = batch_size
        self.use_attention = use_attention
        self.maxseqlen = maxseqlen
        self.maxclauselen = maxclauselen
        self.train = train
        self.input_size = input_size

    def __len__(self):
        return int(np.ceil(len(self.str_seqs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_str_seqs = self.str_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_seqs = self.label_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X, batch_Y = self.make_data(batch_str_seqs, batch_label_seqs)
        return batch_X, batch_Y
    
    def make_data(self,str_seqs, label_seqs):
        if self.train:
            return self.make_data_train(str_seqs, label_seqs)
        else:
            return self.make_data_test(str_seqs)
    
    def make_data_train(self,str_seqs, label_seqs):
        X = []
        Y = []
        Y_inds = []
        
        all_indices = []
        all_segments = []
        para_lens = []
        for str_seq, label_seq in zip(str_seqs, label_seqs):
            if self.use_attention:
                x = np.zeros((self.maxseqlen, self.maxclauselen, self.input_size))
            else:
                x = np.zeros((self.maxseqlen, self.input_size))
            y_ind = np.zeros(self.maxseqlen)
            
            seq_len = len(str_seq)
            if seq_len > self.maxseqlen:
                str_seq = str_seq[:self.maxseqlen]
                seq_len = self.maxseqlen
            
            para_indices = []
            para_segments = []
            for i, (clause, label) in enumerate(zip(str_seq, label_seq)):
                indices, segments = self.tokenizer.encode(clause.lower(), max_len=512)
                para_indices.append(indices)
                para_segments.append(segments)
                
                y_ind[-seq_len+i] = self.label_ind[label]
            
            para_lens.append(len(para_indices))
            all_indices.extend(para_indices)
            all_segments.extend(para_segments)
            Y_inds.append(y_ind)
            
        bert_embedding = self.bert.predict([np.array(all_indices), np.array(all_segments)], batch_size=len(all_indices))[:,:self.maxclauselen,:]
        
        if self.use_attention:
            X = np.zeros((len(str_seqs), self.maxseqlen, self.maxclauselen, self.input_size))
        else:
            X = np.zeros((len(str_seqs), self.maxseqlen, self.input_size))
        
        cumulative_index = 0
        for i, para_len in enumerate(para_lens):
            if self.use_attention:
                X[i,-para_len:,:,:] = bert_embedding[cumulative_index: cumulative_index+para_len,:,:]
            else:
                #x[-seq_len:,:] = np.max(bert_embedding[cumulative_index: cumulative_index+para_len,:,:],axis=1)
                X[i,-para_len:,:] = bert_embedding[cumulative_index: cumulative_index+para_len,0,:]
            cumulative_index += para_len

        for y_ind in Y_inds:
            y = np.zeros((self.maxseqlen, len(self.label_ind)))
            for i, y_ind_i in enumerate(y_ind):
                y[i][y_ind_i.astype(int)] = 1
            Y.append(y)
        
        return X, np.asarray(Y) # One-hot representation of labels
    
    def make_data_test(self,str_seqs):
        X = []
        
        all_indices = []
        all_segments = []
        para_lens = []
        for str_seq in str_seqs:
            if self.use_attention:
                x = np.zeros((self.maxseqlen, self.maxclauselen, self.input_size))
            else:
                x = np.zeros((self.maxseqlen, self.input_size))
            
            seq_len = len(str_seq)
            if seq_len > self.maxseqlen:
                str_seq = str_seq[:self.maxseqlen]
                seq_len = self.maxseqlen
            
            para_indices = []
            para_segments = []
            for i, clause in enumerate(str_seq):
                indices, segments = self.tokenizer.encode(clause.lower(), max_len=512)
                para_indices.append(indices)
                para_segments.append(segments)
                            
            para_lens.append(len(para_indices))
            all_indices.extend(para_indices)
            all_segments.extend(para_segments)
            
        bert_embedding = self.bert.predict([np.array(all_indices), np.array(all_segments)], batch_size=len(all_indices))[:,:self.maxclauselen,:]
        
        if self.use_attention:
            X = np.zeros((len(str_seqs), self.maxseqlen, self.maxclauselen, self.input_size))
        else:
            X = np.zeros((len(str_seqs), self.maxseqlen, self.input_size))
        
        cumulative_index = 0
        for i, para_len in enumerate(para_lens):
            if self.use_attention:
                X[i,-para_len:,:,:] = bert_embedding[cumulative_index: cumulative_index+para_len,:,:]
            else:
                #x[-para_len:,:] = np.max(bert_embedding[cumulative_index: cumulative_index+para_len,:,:],axis=1)
                X[i,-para_len:,:] = bert_embedding[cumulative_index: cumulative_index+para_len,0,:]
            cumulative_index += para_len
        
        return X, np.asarray([])