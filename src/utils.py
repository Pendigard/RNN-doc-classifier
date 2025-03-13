
import codecs
import re
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

def embed_sentence(model, sentence):
    words = sentence.split()
    seq = []
    word_not_in_vocab = []
    for word in words:
        word = word.lower().strip()
        if word in model:
            seq.append(model[word])
        else:
            word_not_in_vocab.append(word)
    return np.array(seq), word_not_in_vocab


def preprocess_pres_set(text, model):
    """
    @brief: get rid of punctuation and lower the text
    """
    punctuation = '!".#,$%&()*+/:;<=>?@[\\]^_`{|}~'
    text = text.lower()
    text_processed = ""
    for i in range(len(text)):
        if text[i] in punctuation:
            continue
        if text[i] == "'":
            if i == 0 or i == len(text) - 1:
                continue
            if text[i - 1] == 'c':
                text_processed += "'"
                continue
            else:
                text_processed = text_processed[:-1] + " "
                continue
        text_processed += text[i]
    word_list = text_processed.split()
    for i in range(len(word_list)):
        if '-' in word_list[i] and word_list[i] not in model:
            tmp_words = word_list[i].split('-')
            word_list.pop(i)
            for j in range(len(tmp_words)):
                word_list.insert(i + j, tmp_words[j])
    text_processed = ' '.join(word_list)
    return text_processed.strip()

def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else:
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs

def get_doc_vec(model, alltxts, alllabs, preprocess=preprocess_pres_set, embed_sentence=embed_sentence):
    allseqs = []
    alllabs_filtered = []
    alltxts_filtered = []
    words_not_in_vocab = []
    avg_word_not_in_vocab = 0
    
    for i in range(len(alltxts)):
        seq, word_not_in_vocab = embed_sentence(model, preprocess(alltxts[i], model))
        avg_word_not_in_vocab += len(word_not_in_vocab)
        
        if len(seq) > 0:
            allseqs.append(seq)
            alllabs_filtered.append(alllabs[i])
            alltxts_filtered.append(alltxts[i])
        
        words_not_in_vocab += word_not_in_vocab
    
    # Convert to PyTorch tensors
    allseqs = [torch.tensor(seq) for seq in allseqs]
    alllabs_filtered = [torch.tensor([1.0]) if lab == 1 else torch.tensor([0.0]) for lab in alllabs_filtered]
    
    return allseqs, alllabs_filtered, alltxts_filtered, words_not_in_vocab


def evaluate(model, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    model.eval()  # Set model to evaluation mode

    def compute_metrics(X, y):
        res = []
        with torch.no_grad():
            for seq in X:
                seq = seq.float().unsqueeze(0)
                y_pred = model(seq)
                y_pred = torch.sigmoid(y_pred)
                res.append(y_pred)
        res = np.array(res).flatten()
        res = [1 if x.item() >= 0.5 else 0 for x in res]
        y = [x.item() for x in y]
        accuracy = accuracy_score(y, res)
        precision = precision_score(y, res)
        recall = recall_score(y, res)
        f1 = f1_score(y, res)
        f1_a = f1_score(y, res, pos_label=0)
        f1_b = f1_score(y, res, pos_label=1)
        return accuracy, precision, recall, f1, f1_a, f1_b

    # Compute validation metrics (if provided)
    if X_val is not None and y_val is not None:
        val_acc, val_prec, val_rec, val_f1, val_f1_a, val_f1_b = compute_metrics(X_val, y_val)
        print(f'Validation - Accuracy: {val_acc}, Precision: {val_prec}, Recall: {val_rec}, F1: {val_f1}, F1_A: {val_f1_a}, F1_B: {val_f1_b}')
    else:
        val_acc = val_prec = val_rec = val_f1 = None

    # Compute test metrics (if provided)
    if X_test is not None and y_test is not None:
        test_acc, test_prec, test_rec, test_f1, test_f1_a, test_f1_b = compute_metrics(X_test, y_test)
        print(f'Test - Accuracy: {test_acc}, Precision: {test_prec}, Recall: {test_rec}, F1: {test_f1}, F1_A: {test_f1_a}, F1_B: {test_f1_b}')
    else:
        test_acc = test_prec = test_rec = test_f1 = None
        # Compute training metrics
    train_acc, train_prec, train_rec, train_f1, train_f1_a, train_f1_b = compute_metrics(X_train, y_train)
    print(f'Training - Accuracy: {train_acc}, Precision: {train_prec}, Recall: {train_rec}, F1: {train_f1}, F1_A: {train_f1_a}, F1_B: {train_f1_b}')
    return {
        "train": (train_acc, train_prec, train_rec, train_f1),
        "val": (val_acc, val_prec, val_rec, val_f1),
        "test": (test_acc, test_prec, test_rec, test_f1),
    }