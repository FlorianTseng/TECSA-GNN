import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.data import DataLoader
from torch.utils.data import random_split, Subset
from datasets import CrystalDataset
from model import CrystalGNN
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

USE_ALL_SAMPLES = False
N = 8192
BATCH_SIZE = 128

LR = 0.0001
EPOCHS = 2500

prop_csv_path = '/project2/zengyuxuan/4thpaper/model/dataset/kappa_prop.csv'

full_dataset = CrystalDataset(
    prop_csv=prop_csv_path,
    cif_file_path='/project2/zengyuxuan/4thpaper/model/cif_files',
    target_name='300',
    elem_feat_path='/project2/zengyuxuan/4thpaper/model/dataset/elem-feat-norm.csv',
    global_feat_path='/project2/zengyuxuan/4thpaper/model/dataset/global-feat-norm.csv'
)

if USE_ALL_SAMPLES:
    dataset = full_dataset
    BATCH_SIZE = max(32, len(dataset) // 10)
else:
    N = min(N, len(full_dataset))
    dataset = Subset(full_dataset, list(range(N)))

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
split_gen = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

sample = full_dataset[0]
global_dim = sample.global_features.shape[0]
node_dim = sample.x.shape[1]
edge_dim = sample.edge_attr.shape[1]
line_dim = sample.line_edge_attr.shape[1]

model = CrystalGNN(node_dim, edge_dim, line_dim, global_dim)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train():
    model.train()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            line_edge_index=data.line_edge_index,
            line_edge_attr=data.line_edge_attr,
            batch=data.batch,
            global_feat=data.global_features
        )

        y = data.y.view(-1)
        loss = criterion(out.view(-1), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        y_true_all.append(y.detach().cpu())
        y_pred_all.append(out.detach().cpu().view(-1))

    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()
    r2 = r2_score(y_true_all, y_pred_all)
    return total_loss / len(train_loader.dataset), r2

def validate():
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)

            out = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                line_edge_index=data.line_edge_index,
                line_edge_attr=data.line_edge_attr,
                batch=data.batch,
                global_feat=data.global_features
            )

            y = data.y.view(-1)
            loss = criterion(out.view(-1), y)
            total_loss += loss.item() * data.num_graphs

            y_true_all.append(y.cpu())
            y_pred_all.append(out.cpu().view(-1))

    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()
    r2 = r2_score(y_true_all, y_pred_all)
    return total_loss / len(val_loader.dataset), r2

best_val_r2 = -float('inf')
log_file = 'kappa-training-log.txt'

with open(log_file, 'w') as f:
    f.write("Epoch | Train Loss | Train R² | Val Loss | Val R²\n")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_r2 = train()
    val_loss, val_r2 = validate()

    log_msg = (f"Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
               f"R²: {train_r2:.4f} | Val Loss: {val_loss:.4e} | R²: {val_r2:.4f}")
    
    print(log_msg)

    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

    if val_r2 > best_val_r2 and train_r2 > val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Saved best model at epoch {epoch} with Val R² = {val_r2:.4f}")

        model.eval()
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    line_edge_index=data.line_edge_index,
                    line_edge_attr=data.line_edge_attr,
                    batch=data.batch,
                    global_feat=data.global_features
                )

                y = data.y.view(-1)
                y_true_all.append(y.cpu())
                y_pred_all.append(out.cpu().view(-1))

        df_train = pd.DataFrame({
            "true": torch.cat(y_true_all).numpy(),
            "pred": torch.cat(y_pred_all).numpy()
        })
        df_train.to_csv("train_results.csv", index=False)
        print("Exported train_results.csv")

        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    line_edge_index=data.line_edge_index,
                    line_edge_attr=data.line_edge_attr,
                    batch=data.batch,
                    global_feat=data.global_features
                )

                y = data.y.view(-1)
                y_true_val.append(y.cpu())
                y_pred_val.append(out.cpu().view(-1))

        df_val = pd.DataFrame({
            "true": torch.cat(y_true_val).numpy(),
            "pred": torch.cat(y_pred_val).numpy()
        })
        df_val.to_csv("val_results.csv", index=False)
        print("Exported val_results.csv")
