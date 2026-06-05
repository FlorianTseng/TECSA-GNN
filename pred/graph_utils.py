from pymatgen.core import Structure
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

def structure_to_graph_with_k_neighbors(structure, k=8, elem_features_df=None):
    atom_types = np.array([site.specie.number for site in structure], dtype=np.int64)
    coords = np.array([site.coords for site in structure], dtype=np.float32)
    lattice = structure.lattice.matrix

    if elem_features_df is None:
        elem_features_df = pd.read_csv('dataset/elem-feat-norm.csv')  # 如果没传，就读一次

    unique_elements = set(site.specie.symbol for site in structure)
    elem_features = {}
    for elem in unique_elements:
        elem_row = elem_features_df[elem_features_df['formula'] == elem]
        if not elem_row.empty:
            elem_features[elem] = elem_row.iloc[0, 1:].values.astype(np.float32)

    all_coords, all_indices = [], []
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                shift = np.dot([a, b, c], lattice)
                for i, site in enumerate(structure):
                    all_coords.append(site.coords + shift)
                    all_indices.append(i)

    all_coords = np.array(all_coords)
    all_indices = np.array(all_indices)

    tree = cKDTree(all_coords)
    edge_index, edge_attr = [], []

    for i, center in enumerate(coords):
        distances, neighbor_ids = tree.query(center, k=k + 1)
        for dist, nid in zip(distances[1:], neighbor_ids[1:]):
            j = all_indices[nid]
            edge_index.append([i, j])
            edge_attr.append([dist])

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)

    return {
        'num_atoms': len(atom_types),
        'atom_types': atom_types,
        'coords': coords,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'lattice': lattice,
        'elem_features': elem_features
    }

def structure_to_line_graph(graph_dict):
    edge_index = graph_dict['edge_index']
    coords = graph_dict['coords']

    line_edge_index, line_edge_attr = [], []

    edge_list = list(zip(edge_index[0], edge_index[1]))
    edge_to_idx = {e: idx for idx, e in enumerate(edge_list)}

    for i, j in edge_list:
        neighbors = edge_index[1][edge_index[0] == j]
        for k in neighbors:
            if k == i:
                continue
            edge1 = (i, j)
            edge2 = (j, k)
            idx1 = edge_to_idx.get(edge1)
            idx2 = edge_to_idx.get(edge2)
            if idx1 is None or idx2 is None:
                continue
            vec1 = coords[i] - coords[j]
            vec2 = coords[k] - coords[j]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            line_edge_index.append([idx1, idx2])
            line_edge_attr.append([angle])

    line_edge_index = np.array(line_edge_index).T
    line_edge_attr = np.array(line_edge_attr)

    return {
        'line_edge_index': line_edge_index,
        'line_edge_attr': line_edge_attr,
        'num_line_nodes': len(edge_list)
    }

def rbf_expand_distances_auto(graph_dict, num_centers=8, cutoff=8.0):
    edge_index = graph_dict['edge_index']
    edge_attr = graph_dict['edge_attr'].flatten()
    dist_centers = np.linspace(0, cutoff, num_centers)
    dist_sigma = (dist_centers[1] - dist_centers[0]) * 1.5
    edge_attr_rbf = np.exp(-((edge_attr[:, None] - dist_centers[None, :]) ** 2) / (dist_sigma ** 2))
    return {
        'edge_index': edge_index,
        'edge_attr_rbf': edge_attr_rbf
    }

def rbf_expand_angles_auto(line_graph_dict, num_centers=8):
    line_edge_index = line_graph_dict['line_edge_index']
    line_edge_attr = line_graph_dict['line_edge_attr'].flatten()
    angle_centers = np.linspace(0, np.pi, num_centers)
    angle_sigma = (angle_centers[1] - angle_centers[0]) * 1.5
    line_edge_attr_rbf = np.exp(-((line_edge_attr[:, None] - angle_centers[None, :]) ** 2) / (angle_sigma ** 2))
    return {
        'line_edge_index': line_edge_index,
        'line_edge_attr_rbf': line_edge_attr_rbf
    }
