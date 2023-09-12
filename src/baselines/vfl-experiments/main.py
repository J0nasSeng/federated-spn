# TODO: add function to get vertically split dataset
# TODO: training loop for ViT and TabNet
from datasets.utils import get_vertical_data
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from models import TabNet
import argparse
from sklearn.metrics import accuracy_score, f1_score
from math import ceil

# hyperparameters from TabNet paper: https://arxiv.org/pdf/1908.07442.pdf
tabnet_hyperparams = {
    'income': {
        'cat_idxs': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'cat_dims': [74, 9, 16, 16, 7, 15, 6, 5, 2, 109, 52, 94, 42],
        'cat_emb_dim': [1] * 13,
        'n_a': 16,
        'n_d': 16,
        'epsilon': 0.0001,
        'gamma': 1.5,
        'n_steps': 5,
        'virtual_batch_size': 1024,
        'grouped_features': [[0, 1, 2], [8, 9, 10]]
        #'grouped_features': [list(range(14))]
    },

    'breast-cancer': {
        'cat_idxs': [],
        'cat_dims': [],
        'cat_emb_dim': 1,
        'n_a': 26,
        'n_d': 24,
        'epsilon': 0.000001,
        'gamma': 1.5,
        'n_steps': 5,
        'virtual_batch_size': 128,
        'grouped_features': [list(range(30))]
    },

    'credit': {
        'cat_idxs': [1, 2, 5, 6, 7, 8, 9],
        'cat_dims': [84, 16, 58, 19, 28, 13, 13],
        'cat_emb_dim': [1, 1, 1, 1, 1, 1, 1],
        'n_a': 26,
        'n_d': 24,
        'epsilon': 0.001,
        'gamma': 1.5,
        'n_steps': 5,
        'virtual_batch_size': 512,
        'grouped_features': [[1, 2, 5, 6, 7, 8, 9], [0, 3, 4]]
    }
}

def get_hyperparameters(subspaces, ds):
    hyperparam_dict = tabnet_hyperparams[ds]
    new_cat_idxs, new_cat_dims, new_grouped_features, new_cat_embed = [], [], [], []
    for client_space in subspaces:
        cat_idxs, cat_size = [], []
        grouped_features = []
        for cat_idx, size in zip(hyperparam_dict['cat_idxs'], hyperparam_dict['cat_dims']):
            if cat_idx in client_space:
                cat_idxs.append(cat_idx)
                cat_size.append(size)
        
        for i, gf in enumerate(hyperparam_dict['grouped_features']):
            group = [g for g in gf if g in client_space]
            if len(group) > 0:
                grouped_features.append(group)
        
        # filter empty grouped features
        grouped_features = [gf for gf in grouped_features if len(gf) > 0]
        # assign new index to each feature as we have mutliple clients
        feature_idx_map = {idx: nidx for nidx, idx in enumerate(client_space)}
        for gf in grouped_features:
            for i in range(len(gf)):
                gf[i] = feature_idx_map[gf[i]]
        for i in range(len(cat_idxs)):
            cat_idxs[i] = feature_idx_map[cat_idxs[i]]
        
        new_cat_dims.append(cat_size)
        new_cat_idxs.append(cat_idxs)
        new_cat_embed.append([1]*len(cat_idxs))
        new_grouped_features.append(grouped_features)
    hyperparam_dict['cat_dims'] = new_cat_dims
    hyperparam_dict['cat_idxs'] = new_cat_idxs
    hyperparam_dict['cat_emb_dim'] = new_cat_embed
    hyperparam_dict['grouped_features'] = new_grouped_features
    print(hyperparam_dict)
    return hyperparam_dict

def main(args):
    train_features, train_targets, test_features, test_targets, subspaces = get_vertical_data(args.dataset, args.num_clients)

    num_train_batches = int(ceil(len(train_features[0]) / args.batch_size))
    num_test_batches = int(ceil(len(test_features[0]) / args.batch_size))

    in_dims = [len(s) for s in subspaces]
    hyperparam_dict = get_hyperparameters(subspaces, args.dataset)
    model = TabNet(args.num_clients, in_dims, 2, **hyperparam_dict)

    train_sets = [TensorDataset(x, train_targets) for x in train_features]
    train_loaders = [DataLoader(ts, args.batch_size) for ts in train_sets]

    test_sets = [TensorDataset(x, test_targets) for x in test_features]
    test_loaders = [DataLoader(ts, args.batch_size) for ts in test_sets]
    
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, args.gamma)

    # train
    for e in range(args.epochs):
        tt_loss = train_epoch(model, train_loaders, optim, criterion, scheduler, num_train_batches)
        avg_acc, avg_f1_micro, avg_f1_macro, val_loss = validate_epoch(model, test_loaders, criterion, num_test_batches)

        print(f"Epoch: {e}, Loss: {tt_loss:.4f}, Val-Loss: {val_loss:.4f}, Accuracy: {avg_acc:.4f}, F1-Micro: {avg_f1_micro:.4f}, F1-Macro: {avg_f1_macro:.4f}")

def train_epoch(model, train_loaders, optim, criterion, scheduler, num_train_batches):
    tt_loss = 0.0
    model.train()
    loader_iters = [iter(l) for l in train_loaders]
    for b in range(num_train_batches):
        x = []
        targets = None
        for tl in loader_iters:
            x_, targets = next(tl)
            x_ = x_.to(dtype=torch.float32)
            x.append(x_)
        optim.zero_grad()
        out, M_loss = model(x)
        #out = torch.softmax(out, dim=1)
        _, predicted = torch.max(out, 1)
        labels_ = F.one_hot(targets).to(torch.float32).squeeze()
        loss = criterion(out, targets.flatten().to(torch.long))
        loss = loss - 1e-3*M_loss
        loss.backward()
        optim.step()
        scheduler.step()
        tt_loss += loss.item() / num_train_batches
    return tt_loss

def validate_epoch(model, test_loaders, criterion, num_test_batches):
    avg_acc, avg_f1_micro, avg_f1_macro, val_loss = 0, 0, 0, 0
    loader_iters = [iter(l) for l in test_loaders]
    for b in range(num_test_batches):
        x = []
        targets = None
        for tl in loader_iters:
            x_, targets = next(tl)
            x_ = x_.to(dtype=torch.float32)
            x.append(x_)
        with torch.no_grad():
            out, M_loss = model(x)
            #out = torch.softmax(out, dim=1)
            labels_ = F.one_hot(targets).to(torch.float32).squeeze()
            _, predicted = torch.max(out, 1)
            _, y_true = torch.max(labels_, 1)
            loss = criterion(out, targets.flatten().to(torch.long))
            loss = loss - 1e-3*M_loss
            val_loss += loss.item() / num_test_batches
            f1_micro = f1_score(y_true.numpy(), predicted.numpy(), average='micro')
            f1_macro = f1_score(y_true.numpy(), predicted.numpy(), average='macro')
            acc = accuracy_score(y_true.numpy(), predicted.numpy())
            avg_acc += acc / num_test_batches
            avg_f1_micro += f1_micro / num_test_batches
            avg_f1_macro += f1_macro / num_test_batches
    return avg_acc, avg_f1_micro, avg_f1_macro, val_loss

parser = argparse.ArgumentParser()

parser.add_argument('--num-clients', default=2, type=int)
parser.add_argument('--dataset', default='income')
parser.add_argument('--gpu', default=None, type=int or None)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=4096, type=int)
parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
#parser.add_argument('--model', default='vit')

args = parser.parse_args()

main(args)