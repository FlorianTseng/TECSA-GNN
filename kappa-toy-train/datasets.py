import torch
import pandas as pd
from pymatgen.core import Structure
from graph_utils import structure_to_graph_with_k_neighbors, structure_to_line_graph, rbf_expand_distances_auto, rbf_expand_angles_auto
import os
import numpy as np
from torch_geometric.data import Data

class CrystalDataset(torch.utils.data.Dataset):
    def __init__(self, prop_csv='dataset/s_prop.csv', cif_file_path='cif_files', target_name='300',
                 elem_feat_path='dataset/elem-feat-norm.csv', global_feat_path='dataset/global-feat-norm.csv'):
        super().__init__()

        self.prop_csv = prop_csv
        self.cif_file_path = cif_file_path
        self.target_name = target_name

        self.data = pd.read_csv(self.prop_csv)
        self.temperature_cols = [str(t) for t in range(100, 1400, 100)]
        self.data['type'] = self.data['type'].map({'n': 0, 'p': 1})

        self.elem_features_df = pd.read_csv(elem_feat_path)
        self.global_features_df = pd.read_csv(global_feat_path)

        self.cif_map = {
            row['mpid']: os.path.join(self.cif_file_path, row['mpid'] + ".cif")
            for _, row in self.data.iterrows()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cif_path = self.cif_map[row['mpid']]
        structure = Structure.from_file(cif_path)

        graph = structure_to_graph_with_k_neighbors(
            structure,
            k=12,
            elem_features_df=self.elem_features_df
        )
        edge_info = rbf_expand_distances_auto(graph)

        edge_index = torch.tensor(edge_info['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(edge_info['edge_attr_rbf'], dtype=torch.float)

        edge_pairs = edge_index.t().tolist()
        reverse_pairs = [[j, i] for i, j in edge_pairs]
        all_pairs = edge_pairs + reverse_pairs
        all_attrs = torch.cat([edge_attr, edge_attr], dim=0)

        unique_pairs = {}
        for pair, attr in zip(all_pairs, all_attrs):
            key = tuple(sorted(pair))
            if key not in unique_pairs:
                unique_pairs[key] = attr

        edge_index = torch.tensor(list(unique_pairs.keys()), dtype=torch.long).t()
        edge_attr = torch.stack(list(unique_pairs.values()))

        line_graph = structure_to_line_graph(graph)
        line_edge_info = rbf_expand_angles_auto(line_graph)

        line_edge_index = torch.tensor(line_edge_info['line_edge_index'], dtype=torch.long)
        line_edge_attr = torch.tensor(line_edge_info['line_edge_attr_rbf'], dtype=torch.float)

        line_pairs = line_edge_index.t().tolist()
        reverse_line_pairs = [[j, i] for i, j in line_pairs]
        all_line_pairs = line_pairs + reverse_line_pairs
        all_line_attrs = torch.cat([line_edge_attr, line_edge_attr], dim=0)

        unique_line_pairs = {}
        for pair, attr in zip(all_line_pairs, all_line_attrs):
            key = tuple(sorted(pair))
            if key not in unique_line_pairs:
                unique_line_pairs[key] = attr

        line_edge_index = torch.tensor(list(unique_line_pairs.keys()), dtype=torch.long).t()
        line_edge_attr = torch.stack(list(unique_line_pairs.values()))

        atom_types = np.array([site.specie.symbol for site in structure])
        atom_features = []
        for elem in atom_types:
            if elem in graph['elem_features']:
                atom_features.append(graph['elem_features'][elem])
            else:
                atom_features.append(np.zeros(len(next(iter(graph['elem_features'].values())))))
        atom_features = np.vstack(atom_features)

        mpid = row['mpid']
        global_features_row = self.global_features_df[self.global_features_df['mpid'] == mpid]

        if global_features_row.empty:
            raise ValueError(f"[ERROR] Global features not found for mpid: {mpid}")

        global_features = global_features_row.drop(columns=['formula', 'mpid']).values[0]

        dope = row['dope']
        type_value = row['type']  # 0 for 'n', 1 for 'p'
        
        global_features = np.concatenate([global_features, [dope, type_value]])

        global_features = torch.tensor(global_features, dtype=torch.float32)

        target = np.array([row[self.target_name]], dtype=np.float32)

        data = Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            line_edge_index=line_edge_index,
            line_edge_attr=line_edge_attr,
            global_features=global_features,
            y=torch.tensor(target, dtype=torch.float)
        )

        return data











if __name__ == '__main__':
    dataset = CrystalDataset()
    count = 0
    idx = 0
    max_samples = len(dataset)

    failed_mpids = []

    while count < max_samples and idx < len(dataset):
        try:
            data = dataset[idx]
            if data is None:
                failed_mpids.append(dataset.data.iloc[idx]['mpid'])
                idx += 1
                continue

            if data.line_edge_attr is None or data.line_edge_attr.size(0) == 0:
                raise ValueError("line_edge_attr is empty")

            print(f"\n=== Sample {idx} (mpid: {dataset.data.iloc[idx]['mpid']}) ===")
            print("x (node features):", data.x.shape)
            print("edge_index:", data.edge_index.shape)
            print("edge_attr:", data.edge_attr.shape)
            print("line_edge_index:", data.line_edge_index.shape)
            print("line_edge_attr:", data.line_edge_attr.shape)
            print("global_features:", data.global_features.shape)
            print("y:", data.y)

            count += 1
        except Exception as e:
            failed_mpid = dataset.data.iloc[idx]['mpid']
            print(f"❌ Error at index {idx} (mpid={failed_mpid}): {e}")
            failed_mpids.append(failed_mpid)
        idx += 1

    print(f"\n✅ 总共成功加载了 {count} 个样本。")
    print(f"❌ 将删除 {len(failed_mpids)} 个失败样本：{failed_mpids}")

    if failed_mpids:
        for mpid in failed_mpids:
            cif_path = os.path.join(dataset.cif_file_path, f"{mpid}.cif")
            if os.path.exists(cif_path):
                os.remove(cif_path)
                print(f"已删除: {cif_path}")

        prop_df = pd.read_csv(dataset.prop_csv)
        prop_df = prop_df[~prop_df['mpid'].isin(failed_mpids)]
        prop_df.to_csv(dataset.prop_csv, index=False)
        print(f"已更新: {dataset.prop_csv}")

        global_feat_path = 'dataset/global-feat-norm.csv'
        global_df = pd.read_csv(global_feat_path)
        global_df = global_df[~global_df['mpid'].isin(failed_mpids)]
        global_df.to_csv(global_feat_path, index=False)
        print(f"已更新: {global_feat_path}")

















