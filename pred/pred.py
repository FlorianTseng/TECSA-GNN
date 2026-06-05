import torch
import pandas as pd
from torch_geometric.data import DataLoader
from datasets import CrystalDataset
from model import CrystalGNN
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
TARGET_TEMP = "300"

prop_csv = r"/project2/zengyuxuan/4thpaper/new-mater/pred/dataset/prop-expanded.csv"
cif_dir = r"/project2/zengyuxuan/4thpaper/new-mater/pred/cif_files"
global_feat_path = r"/project2/zengyuxuan/4thpaper/new-mater/pred/dataset/global-feat-norm.csv"
elem_feat_path = r"/project2/zengyuxuan/4thpaper/new-mater/pred/dataset/elem-feat-norm.csv"
s_model_path = "s-model.pth"
sigma_model_path = "sigma-model.pth"
kappa_model_path = "kappa-model.pth"

dataset = CrystalDataset(
    prop_csv=prop_csv,
    cif_file_path=cif_dir,
    target_name=TARGET_TEMP,
    elem_feat_path=elem_feat_path,
    global_feat_path=global_feat_path
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

sample = dataset[0]
global_dim = sample.global_features.shape[0]
node_dim = sample.x.shape[1]
edge_dim = sample.edge_attr.shape[1]
line_dim = sample.line_edge_attr.shape[1]

def load_model(model_path):
    model = CrystalGNN(node_dim, edge_dim, line_dim, global_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

s_model = load_model(s_model_path)
sigma_model = load_model(sigma_model_path)
kappa_model = load_model(kappa_model_path)

results = []
start_idx = 0

with torch.no_grad():
    for data in loader:
        data = data.to(device)
        batch_size = data.num_graphs

        s_pred = s_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            line_edge_index=data.line_edge_index,
            line_edge_attr=data.line_edge_attr,
            batch=data.batch,
            global_feat=data.global_features
        ).cpu().numpy()

        log_sigma_pred = sigma_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            line_edge_index=data.line_edge_index,
            line_edge_attr=data.line_edge_attr,
            batch=data.batch,
            global_feat=data.global_features
        ).cpu().numpy()

        log_kappa_pred = kappa_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            line_edge_index=data.line_edge_index,
            line_edge_attr=data.line_edge_attr,
            batch=data.batch,
            global_feat=data.global_features
        ).cpu().numpy()

        batch_df = dataset.data.iloc[start_idx:start_idx + batch_size]
        for i in range(batch_size):
            row = batch_df.iloc[i]
            s_val = np.atleast_1d(s_pred[i]).item()
            log_sigma_val = np.atleast_1d(log_sigma_pred[i]).item()
            log_kappa_val = np.atleast_1d(log_kappa_pred[i]).item()
            results.append({
                "formula": row["formula"],
                "mpid": row["mpid"],
                "type": "p" if row["type"] == 1 else "n",
                "dope": row["dope"]*1e20,
                "gap": row["gap"],
                "S": float(s_val),
                "log_sigma": float(log_sigma_val),
                "sigma": float(10 ** log_sigma_val),
                "log_kappa": float(log_kappa_val),
                "kappa": float(10 ** log_kappa_val)
            })

        start_idx += batch_size

df_results = pd.DataFrame(results)
df_results.to_csv("pred-results.csv", index=False)
print("✅ 预测完成，结果已保存为 pred-results.csv")
