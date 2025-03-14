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
from torch.utils.data import DataLoader, TensorDataset


# Reload modules
importlib.reload(ut)
importlib.reload(LSTM_net)

#%%

model = KeyedVectors.load_word2vec_format("../models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, unicode_errors="ignore")



fname = "../data/AFDpresidentutf8/corpus.tache1.learn.utf8.txt"
alltxts, alllabs = ut.load_pres(fname)
# print(len(alltxts))
allseqs, alllabs, alltxts, words_not_in_vocab = ut.get_doc_vec(model, alltxts, alllabs)

# print(len(allseqs))
# print(len(alltxts))
# print(len(alllabs))
# print(len(words_not_in_vocab))

# %%


def weighted_binary_crossentropy(weight_pos, weight_neg):
    def _weighted_binary_crossentropy(y_true, y_pred):
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

def train(model, X_train, y_train, X_test, y_test, X_val, y_val, class_weights, epochs=10, lr=0.01, batch_size=32, n_splits=5, model_path="../models/LSTM_net/LSTM_net_epoch{}.pkl"):
    model.to(device)  # Move model to GPU if available
    model.train()

    weight_pos, weight_neg = class_weights

    criterion = weighted_binary_crossentropy(weight_pos, weight_neg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_log = []
    eval_log = []
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        loss_log.append([])
        split = 0

        for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_train):
            X_tr = [X_train[i] for i in train_idx]
            X_v = [X_train[i] for i in val_idx]
            y_tr = torch.tensor([y_train[i] for i in train_idx], dtype=torch.float32, device=device)
            y_v = torch.tensor([y_train[i] for i in val_idx], dtype=torch.float32, device=device)
            split += 1

            # Padding sequences for batching
            X_tr_padded = pad_sequence(X_tr, batch_first=True, padding_value=0.0)
            X_v_padded = pad_sequence(X_v, batch_first=True, padding_value=0.0)

            # Création des datasets
            train_dataset = TensorDataset(X_tr_padded, y_tr)
            val_dataset = TensorDataset(X_v_padded, y_v)

            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


            # Training loop
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                y_pred = model(X_batch)
                y_pred = torch.sigmoid(y_pred)

                loss = criterion(y_batch, y_pred.squeeze())
                total_loss += loss.item()
                loss_log[-1].append(loss.item())
                loss.backward()
                optimizer.step()


            # Compute validation loss for this fold
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device).float()
                    y_batch = y_batch.to(device).float()

                    y_pred = model(X_batch)
                    y_pred = torch.sigmoid(y_pred)

                    loss = criterion(y_batch, y_pred.squeeze())
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            print(f'Epoch {epoch} {split}/{n_splits}- Train Loss: {total_loss / len(X_train)}, Val Loss: {avg_val_loss}')

        # Save model checkpoint
        pickle.dump(model, open(model_path.format(epoch), 'wb'))

        # Evaluate model on the test and validation sets
        eval_log.append(ut.evaluate(model, X_train, y_train, X_test, y_test, device))
        torch.cuda.empty_cache()
    return loss_log, eval_log


# Initialize model
net = LSTM_net.LSTM_net(200, 128, 2, 1).to(device)



# Split dataset
X_train, X_test, y_train, y_test = train_test_split(allseqs, alllabs, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    
# weight_pos, weight_neg = compute_class_weights(np.array(y_train).flatten())
class_weights = (1, 5)

# Train model
train(net, X_train, y_train, X_test, y_test, X_val, y_val, class_weights, epochs=5, lr=0.01)

# %%
