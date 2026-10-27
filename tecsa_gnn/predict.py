"""Run the original 300 K checkpoints on prepared CSV/CIF inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from .datasets import CrystalDataset
from .model import CrystalGNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--properties', default='data/s_prop.csv.gz')
    p.add_argument('--cifs', default='data/cif_files')
    p.add_argument('--global-features', default='data/global-feat-norm.csv.gz')
    p.add_argument('--element-features', default='data/elem-feat-norm.csv')
    p.add_argument('--weights', default='weights')
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--limit', type=int, default=None, help='Use the first N input rows')
    args = p.parse_args()
    dataset = CrystalDataset(args.properties, args.cifs, '300', args.element_features, args.global_features)
    if args.limit is not None:
        dataset.data = dataset.data.iloc[:args.limit].reset_index(drop=True)
    if not len(dataset):
        raise ValueError('The input has no rows')
    sample = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for name in ('s', 'sigma', 'kappa'):
        model = CrystalGNN(sample.x.shape[1], sample.edge_attr.shape[1], sample.line_edge_attr.shape[1], sample.global_features.shape[0])
        model.load_state_dict(torch.load(Path(args.weights) / f'{name}-model.pth', map_location=device, weights_only=True))
        models.append(model.to(device).eval())
    results = []
    offset = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            batch = batch.to(device)
            outputs = [m(batch.x, batch.edge_index, batch.edge_attr, batch.line_edge_index, batch.line_edge_attr, batch.batch, batch.global_features).cpu().numpy() for m in models]
            for i, (_, row) in enumerate(dataset.data.iloc[offset:offset + batch.num_graphs].iterrows()):
                s, sigma, kappa = (float(o[i]) for o in outputs)
                results.append({'formula': row['formula'], 'mpid': row['mpid'], 'type': 'p' if row['type'] == 1 else 'n', 'dope_cm-3': row['dope'] * 1e20, 'gap_eV': row['gap'], 'S': s, 'log10_sigma_over_tau': sigma, 'sigma_over_tau': 10 ** sigma, 'log10_kappa_e_over_tau': kappa, 'kappa_e_over_tau': 10 ** kappa})
            offset += batch.num_graphs
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f'Wrote {len(results)} predictions to {args.output}')

if __name__ == '__main__':
    main()
