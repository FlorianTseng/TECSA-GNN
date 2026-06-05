import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter, scatter_softmax

class EdgeAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        feat = torch.cat([x[row], x[col], edge_attr], dim=-1)
        msg = self.mlp(feat)
        alpha = self.att_mlp(msg).squeeze(-1)
        alpha = scatter_softmax(alpha, row)
        out = scatter(alpha.unsqueeze(-1) * msg, row, dim=0, dim_size=x.size(0), reduce='sum')
        out = self.norm(out + x)
        return out

class LineGraphConv(nn.Module):
    def __init__(self, hidden_dim, line_dim, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + line_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_feat, line_edge_index, line_edge_attr):
        src, dst = line_edge_index
        max_e = edge_feat.size(0)
        mask = (src >= 0) & (src < max_e) & (dst >= 0) & (dst < max_e)
        src, dst, l_attr = src[mask], dst[mask], line_edge_attr[mask]
        feat = torch.cat([edge_feat[src], edge_feat[dst], l_attr], dim=-1)
        out = self.mlp(feat)
        agg = scatter(out, dst, dim=0, dim_size=max_e, reduce='mean')
        agg = self.norm(agg + edge_feat)
        return agg

class CrystalGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 line_dim,
                 global_dim,
                 hidden_dim=256,
                 num_layers=5,
                 dropout=0):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.edge_proj_in = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.line_proj = nn.Sequential(
            nn.Linear(line_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'edge_att': EdgeAttention(hidden_dim, dropout),
                'line_conv': LineGraphConv(hidden_dim, hidden_dim, dropout)
            }))

        self.node_pool_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_pool_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_norm = nn.LayerNorm(hidden_dim + global_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_attr, line_edge_index, line_edge_attr, batch, global_feat):
        x = self.node_proj(x)
        e = self.edge_proj_in(edge_attr)
        l = self.line_proj(line_edge_attr)

        for layer in self.layers:
            x = layer['edge_att'](x, edge_index, e)
            e = layer['line_conv'](e, line_edge_index, l)

        node_pool = global_mean_pool(x, batch)
        edge_batch = batch[edge_index[0]]
        num_graphs = int(batch.max().item()) + 1
        edge_pool = global_mean_pool(e, edge_batch, size=num_graphs)

        node_pool_p = self.node_pool_proj(node_pool)
        edge_pool_p = self.edge_pool_proj(edge_pool)
        global_feat = global_feat.view(num_graphs, -1)

        h = torch.cat([node_pool_p, edge_pool_p, global_feat], dim=1)
        h = self.layer_norm(h)
        out = self.mlp(h).view(-1)
        return out
