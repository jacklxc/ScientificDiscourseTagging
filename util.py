import codecs
import numpy
import glob
import re
from sklearn.metrics import f1_score

def read_passages(filename, is_labeled):
    str_seqs = []
    str_seq = []
    label_seqs = []
    label_seq = []
    for line in codecs.open(filename, "r", "utf-8"):
        lnstrp = line.strip()
        if lnstrp == "":
            if len(str_seq) != 0:
                str_seqs.append(str_seq)
                str_seq = []
                label_seqs.append(label_seq)
                label_seq = []
        else:
            if is_labeled:
                clause, label = lnstrp.split("\t")
                label_seq.append(label.strip())
            else:
                clause = lnstrp
            str_seq.append(clause)
    if len(str_seq) != 0:
        str_seqs.append(str_seq)
        str_seq = []
        label_seqs.append(label_seq)
        label_seq = []
    return str_seqs, label_seqs

def from_BIO_ind(BIO_pred, BIO_target, indices):
    table = {} # Make a mapping between the indices of BIO_labels and temporary original label indices
    original_labels = []
    for BIO_label,BIO_index in indices.items():
        if BIO_label[:2] == "I_" or BIO_label[:2] == "B_":
            label = BIO_label[2:]
        else:
            label = BIO_label
        if label in original_labels:
            table[BIO_index] = original_labels.index(label)
        else:
            table[BIO_index] = len(original_labels)
            original_labels.append(label)

    original_pred = [table[label] for label in BIO_pred]
    original_target = [table[label] for label in BIO_target]
    return original_pred, original_target

def to_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        prev = ""
        for label in label_para:
            if label!="none": # "none" is O, remain unchanged.
                if label==prev:
                    new_label = "I_"+label
                else:
                    new_label = "B_"+label
            else:
                new_label = label # "none"
            prev = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def from_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        for label in label_para:
            if label[:2] == "I_" or label[:2] == "B_":
                new_label = label[2:]
            else:
                new_label = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def clean_url(word):
    """
        Clean specific data format from social media
    """
    # clean urls
    word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
    word = re.sub(r'exlink', '<URL>', word)
    return word

def clean_num(word):
    # check if the word contain number and no letters
    if any(char.isdigit() for char in word):
        try:
            num = float(word.replace(',', ''))
            return '@'
        except:
            if not any(char.isalpha() for char in word):
                return '@'
    return word


def clean_words(str_seqs):
    processed_seqs = []
    for str_seq in str_seqs:
        processed_clauses = []
        for clause in str_seq:
            filtered = []
            tokens = clause.split()                 
            for word in tokens:
                word = clean_url(word)
                word = clean_num(word)
                filtered.append(word)
            filtered_clause = " ".join(filtered)
            processed_clauses.append(filtered_clause)
        processed_seqs.append(processed_clauses)
    return processed_seqs

def test_f1(test_file,pred_label_seqs):
    def linearize(labels):
        linearized = []
        for paper in labels:
            for label in paper:
                linearized.append(label)
        return linearized
    _, label_seqs = read_passages_original(test_file,True)
    true_label = linearize(label_seqs)
    pred_label = linearize(pred_label_seqs)

    f1 = f1_score(true_label,pred_label,average="weighted")
    print("F1 score:",f1)
    return f1
    
def evaluate(y, pred):
    accuracy = float(sum([c == p for c, p in zip(y, pred)]))/len(pred)
    num_gold = {}
    num_pred = {}
    num_correct = {}
    for c, p in zip(y, pred):
        if c in num_gold:
            num_gold[c] += 1
        else:
            num_gold[c] = 1
        if p in num_pred:
            num_pred[p] += 1
        else:
            num_pred[p] = 1
        if c == p:
            if c in num_correct:
                num_correct[c] += 1
            else:
                num_correct[c] = 1
    fscores = {}
    for p in num_pred:
        precision = float(num_correct[p]) / num_pred[p] if p in num_correct else 0.0
        recall = float(num_correct[p]) / num_gold[p] if p in num_correct else 0.0
        fscores[p] = 2 * precision * recall / (precision + recall) if precision !=0 and recall !=0 else 0.0
    weighted_fscore = sum([fscores[p] * num_gold[p] if p in num_gold else 0.0 for p in fscores]) / sum(num_gold.values())
    return accuracy, weighted_fscore, fscores

def make_folds(train_X, train_Y, num_folds):
    num_points = train_X.shape[0]
    fol_len = num_points / num_folds
    rem = num_points % num_folds
    print(train_X.shape, train_Y.shape)
    X_folds = numpy.split(train_X, num_folds) if rem == 0 else numpy.split(train_X[:-rem], num_folds)
    Y_folds = numpy.split(train_Y, num_folds) if rem == 0 else numpy.split(train_Y[:-rem], num_folds)
    cv_folds = []
    for i in range(num_folds):
        train_folds_X = []
        train_folds_Y = []
        for j in range(num_folds):
            if i != j:
                train_folds_X.append(X_folds[j])
                train_folds_Y.append(Y_folds[j])
        train_fold_X = numpy.concatenate(train_folds_X)
        train_fold_Y = numpy.concatenate(train_folds_Y)
        cv_folds.append(((train_fold_X, train_fold_Y), (X_folds[i], Y_folds[i])))
    return cv_folds

def arg2param(args):
    params = vars(args)
    params["lr"] = float(args.lr)
    params["hard_k"] = int(args.hard_k)
    params["embedding_dropout"] = float(args.embedding_dropout)
    params["high_dense_dropout"] = float(args.high_dense_dropout)
    params["attention_dropout"] = float(args.attention_dropout)
    params["lstm_dropout"] = float(args.lstm_dropout)
    params["word_proj_dim"] = int(args.word_proj_dim)
    params["lstm_dim"] = int(args.lstm_dim)
    params["att_proj_dim"] = int(args.att_proj_dim)
    params["rec_hid_dim"] = int(args.rec_hid_dim)
    params["epoch"] = int(args.epoch)
    params["maxseqlen"] = int(args.maxseqlen)
    params["maxclauselen"] = int(args.maxclauselen)
    params["batch_size"]=int(args.batch_size)
    params["validation_split"] = float(args.validation_split)
    return params
