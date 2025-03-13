#%%
import importlib
from gensim.models import KeyedVectors
import numpy as np
import LSTM_net
import torch
from sklearn.model_selection import train_test_split, KFold
import utils as ut
from tqdm import tqdm
import pickle
from torch.nn.utils.rnn import pad_sequence

importlib.reload(ut)

#%%

model = KeyedVectors.load_word2vec_format("../models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, unicode_errors="ignore")



fname = "../data/AFDpresidentutf8/corpus.tache1.learn.utf8.txt"
alltxts, alllabs = ut.load_pres(fname)
print(len(alltxts))
allseqs, alllabs, alltxts, words_not_in_vocab = ut.get_doc_vec(model, alltxts, alllabs)

print(len(allseqs))
print(len(alltxts))
print(len(alllabs))
print(len(words_not_in_vocab))

# %%


def weighted_binary_crossentropy(weight_pos, weight_neg):
    def _weighted_binary_crossentropy(y_true, y_pred):
        y_pred = y_pred.unsqueeze(0)  # Ensure the correct shape
        # Compute the binary cross-entropy
        b_ce = torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction='none')
        # Apply weights based on class
        weight_vector = y_true * weight_pos + (1. - y_true) * weight_neg
        weighted_b_ce = weight_vector * b_ce
        return torch.mean(weighted_b_ce)  # Return the mean loss
    return _weighted_binary_crossentropy

def compute_class_weights(y_train):
    """Compute class weights dynamically."""
    class_counts = np.bincount(y_train)  # Count occurrences of each class
    total_samples = len(y_train)
    
    weight_neg = total_samples / (2.0 * class_counts[0])  # Weight for class 0
    weight_pos = total_samples / (2.0 * class_counts[1])  # Weight for class 1

    return weight_pos, weight_neg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(model, X_train, y_train, X_test, y_test, X_val, y_val, epochs=10, lr=0.01, n_splits=5, model_path="../models/LSTM_net/LSTM_net_epoch{}.pkl"):
    model.to(device)  # Move model to GPU if available
    model.train()
    
    weight_pos, weight_neg = compute_class_weights(y_train)
    criterion = weighted_binary_crossentropy(weight_pos, weight_neg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_log = []
    eval_log = []
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        loss_log.append([])
        
        fold_val_losses = []  # Store validation losses for this epoch

        for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_train):
            X_tr = [X_train[i] for i in train_idx]
            X_v = [X_train[i] for i in val_idx]
            y_tr = torch.tensor([y_train[i] for i in train_idx], dtype=torch.float32, device=device)
            y_v = torch.tensor([y_train[i] for i in val_idx], dtype=torch.float32, device=device)

            # Padding sequences for batching
            X_tr_padded = pad_sequence(X_tr, batch_first=True, padding_value=0.0).to(device)
            X_v_padded = pad_sequence(X_v, batch_first=True, padding_value=0.0).to(device)

            # Training loop
            for i, seq in enumerate(X_tr_padded):
                optimizer.zero_grad()
                
                seq = seq.float().unsqueeze(0).to(device)  # Move sequence to GPU
                y = y_tr[i].float().unsqueeze(0).to(device)  # Move label to GPU

                y_pred = model(seq)
                y_pred = torch.sigmoid(y_pred)

                loss = criterion(y, y_pred.squeeze())
                loss_log[epoch].append(loss.item())
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Compute validation loss for this fold
            with torch.no_grad():
                val_losses = []
                for seq, y in zip(X_v_padded, y_v):
                    seq = seq.float().unsqueeze(0).to(device)
                    y = y.float().unsqueeze(0).to(device)

                    y_pred = model(seq)
                    y_pred = torch.sigmoid(y_pred)
                    loss = criterion(y, y_pred.squeeze())
                    val_losses.append(loss.item())
                
                fold_val_losses.append(np.mean(val_losses))  # Average loss for this fold

        # Average validation loss across all folds
        avg_val_loss = np.mean(fold_val_losses)
        print(f'Epoch {epoch} - Train Loss: {total_loss / len(X_train)}, Val Loss: {avg_val_loss}')

        # Save model checkpoint
        pickle.dump(model, open(model_path.format(epoch), 'wb'))

        # Evaluate model on the test and validation sets
        eval_log.append(ut.evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test))

    return loss_log, eval_log


# Initialize model
net = LSTM_net.LSTM_net(200, 128, 2, 1).to(device)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(allseqs, alllabs, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Move Data to CUDA
X_train = [seq.to(device) for seq in X_train]
X_test = [seq.to(device) for seq in X_test]
X_val = [seq.to(device) for seq in X_val]
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

# Train model
train(net, X_train, y_train, X_test, y_test, X_val, y_val, epochs=10, lr=0.001)

# %%
