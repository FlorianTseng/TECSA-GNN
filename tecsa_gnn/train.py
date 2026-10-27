"""Train a 300 K model on the supplied prepared data."""
import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from .datasets import CrystalDataset
from .model import CrystalGNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=('s', 'sigma', 'kappa'), default='s')
    p.add_argument('--properties', help='Override default property CSV')
    p.add_argument('--cifs', default='data/cif_files')
    p.add_argument('--global-features', default='data/global-feat-norm.csv.gz')
    p.add_argument('--element-features', default='data/elem-feat-norm.csv')
    p.add_argument('--limit', type=int, default=4096, help='First N rows; 0 uses all rows')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='best_model.pth')
    a = p.parse_args()
    defaults = {'s': ('data/s_prop.csv.gz', 80, 200, 0.01), 'sigma': ('data/sigma_prop-replaced.csv.gz', 128, 2500, 0.0001), 'kappa': ('data/kappa_prop.csv.gz', 128, 2500, 0.0001)}
    file, batch_size, epochs, lr = defaults[a.target]
    batch_size, epochs, lr = a.batch_size or batch_size, a.epochs or epochs, a.lr or lr
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    full = CrystalDataset(a.properties or file, a.cifs, '300', a.element_features, a.global_features)
    dataset = Subset(full, range(min(a.limit, len(full)))) if a.limit else full
    if len(dataset) < 2: raise ValueError('At least two rows are required')
    n = int(0.9 * len(dataset))
    train, val = random_split(dataset, [n, len(dataset)-n], generator=torch.Generator().manual_seed(a.seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    sample = full[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrystalGNN(sample.x.shape[1], sample.edge_attr.shape[1], sample.line_edge_attr.shape[1], sample.global_features.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best = float('inf')
    for epoch in range(1, epochs + 1):
        losses = []
        for loader, training in ((train_loader, True), (val_loader, False)):
            model.train(training); total = 0
            with torch.set_grad_enabled(training):
                for b in loader:
                    b = b.to(device)
                    if training: optimizer.zero_grad()
                    pred = model(b.x, b.edge_index, b.edge_attr, b.line_edge_index, b.line_edge_attr, b.batch, b.global_features)
                    loss = torch.nn.functional.mse_loss(pred.view(-1), b.y.view(-1))
                    if training: loss.backward(); optimizer.step()
                    total += loss.item() * b.num_graphs
            losses.append(total / len(loader.dataset))
        print(f'Epoch {epoch:04d} train MSE={losses[0]:.6g} validation MSE={losses[1]:.6g}', flush=True)
        if losses[1] < best:
            best = losses[1]; torch.save(model.state_dict(), a.output)

if __name__ == '__main__':
    main()
