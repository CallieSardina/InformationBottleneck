# training gnn models, we already shared trained ones at data/{dataset_name}/gnn

import os
import argparse
import torch
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from math import floor
from tqdm import tqdm

import data

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from edge_estimator import estimate_mi

class GNN(torch.nn.Module):

    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0):
        super(GNN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First GCN layer.
        self.convs.append(GCNConv(num_features, dim))
        self.bns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)
        self.fc3 = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def forward(self, data, edge_weight=None):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        embeddings_per_layer = []

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.
            embeddings_per_layer.append(x)

        # Pooling and FCs.
        node_embeddings = x
        out1 = self.fc1(node_embeddings)
        embeddings_per_layer.append(out1)
        out2 = self.fc2(out1)
        embeddings_per_layer.append(out2)
        graph_embedding = global_max_pool(out2, batch)
        out3 = self.fc3(graph_embedding)
        logits = F.log_softmax(out3, dim=-1)

        return node_embeddings, graph_embedding, embeddings_per_layer, logits

def save_activations(activations, epoch, batch_idx, layer_dir='layer_activations_mi_with_fc_layers'):
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    for layer_idx, activation in enumerate(activations):
        file_path = os.path.join(layer_dir, f'epoch_{epoch}_batch_{batch_idx}_layer_{layer_idx}.pt')
        torch.save(activation, file_path)

def train(model, optimizer, train_loader, epoch, device):
    model.train()
    total_loss = 0

    mi_XZ_1s = []
    mi_XZ_2s = []
    mi_XZ_3s = []
    mi_XZ_4s = []
    mi_XZ_5s = []
    mi_ZY_1s = []
    mi_ZY_2s = []
    mi_ZY_3s = []
    mi_ZY_4s = []
    mi_ZY_5s = []
    for batch_idx, train_batch in tqdm(enumerate(train_loader), desc='Train Batch', total=len(train_loader)):
        optimizer.zero_grad()
        node_embeddings, graph_embedding, embeddings_per_layer, logits = model(train_batch.to(device))

        save_activations(embeddings_per_layer, epoch, batch_idx)

        batch_size = len(train_batch)

        emb1 = embeddings_per_layer[0].unsqueeze(0).repeat(batch_size, 1, 1) 
        emb2 = embeddings_per_layer[1].unsqueeze(0).repeat(batch_size, 1, 1) 
        emb3 = embeddings_per_layer[2].unsqueeze(0).repeat(batch_size, 1, 1)
        emb4 = embeddings_per_layer[3].unsqueeze(0).repeat(batch_size, 1, 1)
        emb5 = embeddings_per_layer[4].unsqueeze(0).repeat(batch_size, 1, 1) 

        mi_XZ_1 = estimate_mi(train_batch.x, emb1)
        mi_XZ_2 = estimate_mi(train_batch.x, emb2)
        mi_XZ_3 = estimate_mi(train_batch.x, emb3)
        mi_XZ_4 = estimate_mi(train_batch.x, emb4)
        mi_XZ_5 = estimate_mi(train_batch.x, emb5)

        mi_ZY_1 = estimate_mi(train_batch.y, emb1)
        mi_ZY_2 = estimate_mi(train_batch.y, emb2)
        mi_ZY_3 = estimate_mi(train_batch.y, emb3)
        mi_ZY_4 = estimate_mi(train_batch.y, emb4)
        mi_ZY_5 = estimate_mi(train_batch.y, emb5)

        mi_XZ_1s.append(mi_XZ_1)
        mi_XZ_2s.append(mi_XZ_2)
        mi_XZ_3s.append(mi_XZ_3)
        mi_XZ_4s.append(mi_XZ_4)
        mi_XZ_5s.append(mi_XZ_5)

        mi_ZY_1s.append(mi_ZY_1)
        mi_ZY_2s.append(mi_ZY_2)
        mi_ZY_3s.append(mi_ZY_3)
        mi_ZY_4s.append(mi_ZY_4)
        mi_ZY_5s.append(mi_ZY_5)

        with open('./run_results_with_fc_layers.txt', 'a') as f:
            print(f"Epoch {epoch}, Batch {batch_idx}, MI_XZ: {mi_XZ_1}, {mi_XZ_2}, {mi_XZ_3} , {mi_XZ_4} , {mi_XZ_5}, MI_ZY: {mi_ZY_1}, {mi_ZY_2}, {mi_ZY_3}, {mi_ZY_4}, {mi_ZY_5}", file=f)

        # print(f"Epoch {epoch}, Batch {batch_idx}, MI_XZ: {mi_XZ_1}, {mi_XZ_2}, {mi_XZ_3}, MI_ZY: {mi_ZY}")

        loss = F.nll_loss(logits, train_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * train_batch.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval(model, eval_loader, device):
    model.eval()
    total_loss = 0
    total_hits = 0

    for eval_batch in tqdm(eval_loader, desc='Eval Batch', total=len(eval_loader)):
        logits = model(eval_batch.to(device))[-1]
        loss = F.nll_loss(logits, eval_batch.y)
        total_loss += loss.item() * eval_batch.num_graphs
        pred = torch.argmax(logits, dim=-1)
        hits = (pred == eval_batch.y).sum()
        total_hits += hits

    return total_loss / len(eval_loader.dataset), total_hits / len(eval_loader.dataset)


def split_data(dataset, valid_ratio=0.1, test_ratio=0.1):
    valid_size = floor(len(dataset) * valid_ratio)
    test_size = floor(len(dataset) * test_ratio)
    train_size = len(dataset) - valid_size - test_size
    splits = torch.utils.data.random_split(dataset, lengths=[train_size, valid_size, test_size])

    return splits


def load_trained_gnn(dataset_name, device):
    dataset = data.load_dataset(dataset_name)
    model = GNN(
        num_features=dataset.num_features,
        num_classes=2,
        num_layers=3,
        dim=20,
        dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(f'data/{dataset_name}/gnn_fc/model_best.pth', map_location=device))
    return model


@torch.no_grad()
def load_trained_prediction(dataset_name, device):
    prediction_file = f'data/{dataset_name}/gnn_fc/preds.pt'
    if os.path.exists(prediction_file):
        return torch.load(prediction_file, map_location=device)
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device)
        model.eval()

        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            logits = model(eval_batch.to(device))[-1]
            pred = torch.argmax(logits, dim=-1)
            preds.append(pred)
        preds = torch.cat(preds)
        torch.save(preds, prediction_file)
        return preds


@torch.no_grad()
def load_trained_embeddings_logits(dataset_name, device):
    node_embeddings_file = f'data/{dataset_name}/gnn_fc/node_embeddings.pt'
    graph_embeddings_file = f'data/{dataset_name}/gnn_fc/graph_embeddings.pt'
    logits_file = f'data/{dataset_name}/gnn_fc/logits.pt'
    if os.path.exists(node_embeddings_file) and os.path.exists(graph_embeddings_file) and os.path.exists(logits_file):
        node_embeddings = torch.load(node_embeddings_file)
        for i, node_embedding in enumerate(node_embeddings):  # every graph has different size
            node_embeddings[i] = node_embeddings[i].to(device)
        graph_embeddings = torch.load(graph_embeddings_file, map_location=device)
        logits = torch.load(logits_file, map_location=device)
        return node_embeddings, graph_embeddings, logits
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device)
        model.eval()
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        graph_embeddings, node_embeddings, logits = [], [], []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            node_emb, graph_emb, logit = model(eval_batch.to(device))
            max_batch_number = max(eval_batch.batch)
            for i in range(max_batch_number + 1):
                idx = torch.where(eval_batch.batch == i)[0]
                node_embeddings.append(node_emb[idx])
            graph_embeddings.append(graph_emb)
            logits.append(logit)
        graph_embeddings = torch.cat(graph_embeddings)
        logits = torch.cat(logits)
        torch.save([node_embedding.cpu() for node_embedding in node_embeddings], node_embeddings_file)
        torch.save(graph_embeddings, graph_embeddings_file)
        torch.save(logits, logits_file)
        return node_embeddings, graph_embeddings, logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutagenicity',
                        help="Dataset. Options are ['mutagenicity', 'aids', 'nci1', 'proteins']. Default is 'mutagenicity'. ")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size. Default is 128.')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20,
                        help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for training. Default is 0. ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs. Default is 1000. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--train', type=int, default=0,
                        help='Train=1, just Test=0.')
    return parser.parse_args()

def visualize_embeddings(model, dataloader, device):
    model.eval()
    
    all_labels = []
    all_embeddings = {i: [] for i in range(model.num_layers)}

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, _, embeddings_per_layer, _ = model(data)
            
            for i, emb in enumerate(embeddings_per_layer):
                all_embeddings[i].append(emb.cpu().numpy())

            all_labels.append(data.y.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.concatenate(all_labels)
    for i in range(model.num_layers):
        all_embeddings[i] = np.concatenate(all_embeddings[i])

    # Create subplots
    fig_pca, axes_pca = plt.subplots(1, model.num_layers, figsize=(4 * model.num_layers, 4))
    fig_tsne, axes_tsne = plt.subplots(1, model.num_layers, figsize=(4 * model.num_layers, 4))

    # Apply PCA and t-SNE to each layer's embeddings
    for i in range(model.num_layers):
        embeddings = all_embeddings[i]

        # PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=all_labels, ax=axes_pca[i], palette="Set1")
        axes_pca[i].set_title(f'Layer {i+1} - PCA')
        axes_pca[i].set_xticks([])
        axes_pca[i].set_yticks([])

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)
        sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], hue=all_labels, ax=axes_tsne[i], palette="Set1")
        axes_tsne[i].set_title(f'Layer {i+1} - t-SNE')
        axes_tsne[i].set_xticks([])
        axes_tsne[i].set_yticks([])

    # Adjust layout and show plots
    fig_pca.suptitle('PCA of Embeddings at Each Layer', fontsize=16)
    fig_tsne.suptitle('t-SNE of Embeddings at Each Layer', fontsize=16)
    
    plt.show()

def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load and split the dataset.
    dataset = data.load_dataset(args.dataset)
    train_set, valid_set, test_set = split_data(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Logging.
    gnn_folder = f'data/{args.dataset}/gnn_fc/'
    if not os.path.exists(gnn_folder):
        os.makedirs(gnn_folder)
    log_file = gnn_folder + 'log.txt'
    with open(log_file, 'w') as f:
        pass

    # Initialize the model.
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = GNN(
        num_features=dataset.num_features,
        num_classes=2,
        num_layers=args.num_layers,
        dim=args.dim,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.train == 1:
        # Training.
        start_epoch = 1
        epoch_iterator = tqdm(range(start_epoch, start_epoch + args.epochs), desc='Epoch')
        best_valid = float('inf')
        best_epoch = 0
        for epoch in epoch_iterator:
            train_loss = train(model, optimizer, train_loader, epoch, device)
            valid_loss, valid_acc = eval(model, valid_loader, device)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                print("Best epoch: ", best_epoch)
                torch.save(model.state_dict(), gnn_folder + f'model_checkpoint{epoch}.pth')
                torch.save(optimizer.state_dict(), gnn_folder + f'optimizer_checkpoint{epoch}.pth')
            with open(log_file, 'a') as f:
                print(f'Epoch = {epoch}:', file=f)
                print(f'Train Loss = {train_loss:.4e}', file=f)
                print(f'Valid Loss = {valid_loss:.4e}', file=f)
                print(f'Valid Accuracy = {valid_acc:.4f}', file=f)


    # Testing.
    model.load_state_dict(torch.load(gnn_folder + f'model_checkpoint{best_epoch}.pth', map_location=device))
    torch.save(model.state_dict(), gnn_folder + f'model_best.pth')
    train_acc = eval(model, train_loader, device)[1]
    valid_acc = eval(model, valid_loader, device)[1]
    test_acc = eval(model, test_loader, device)[1]
    with open(log_file, 'a') as f:
        print(file=f)
        print(f"Best Epoch = {best_epoch}:", file=f)
        print(f"Train Accuracy = {train_acc:.4f}", file=f)
        print(f"Valid Accuracy = {valid_acc:.4f}", file=f)
        print(f"Test Accuracy = {test_acc:.4f}", file=f)

    # visualize_embeddings(model, test_loader, device)


if __name__ == '__main__':
    main()