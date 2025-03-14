
import codecs
import re
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

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


def evaluate(model, X_train, y_train, X_test, y_test, device, batch_size=32):
    model.to(device)  # Move model to GPU
    model.eval()  # Set model to evaluation mode

    def compute_metrics(X, y):
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        true_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                y_pred = model(X_batch)  # (batch_size, 1)
                y_pred = torch.sigmoid(y_pred).cpu().numpy().flatten()  # Convertir en numpy

                preds.extend((y_pred >= 0.5).astype(int))  # Convertir en 0/1
                true_labels.extend(y_batch.cpu().numpy())  # Liste des vraies valeurs

        # Calcul des métriques
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        f1_a = f1_score(true_labels, preds, pos_label=0, zero_division=0)
        f1_b = f1_score(true_labels, preds, pos_label=1, zero_division=0)

        return accuracy, precision, recall, f1, f1_a, f1_b

    # Convertir en tensors
    X_train_tensor = pad_sequence(X_train, batch_first=True, padding_value=0.0)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    X_test_tensor = pad_sequence(X_test, batch_first=True, padding_value=0.0)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Évaluation sur test
    test_acc, test_prec, test_rec, test_f1, test_f1_a, test_f1_b = compute_metrics(X_test_tensor, y_test_tensor)
    print(f'Test - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}, F1_A: {test_f1_a:.4f}, F1_B: {test_f1_b:.4f}')

    # Évaluation sur train
    train_acc, train_prec, train_rec, train_f1, train_f1_a, train_f1_b = compute_metrics(X_train_tensor, y_train_tensor)
    print(f'Training - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, F1_A: {train_f1_a:.4f}, F1_B: {train_f1_b:.4f}')

    return {
        "train": (train_acc, train_prec, train_rec, train_f1),
        "test": (test_acc, test_prec, test_rec, test_f1)
    }