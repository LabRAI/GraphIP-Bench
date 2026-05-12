import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import dgl
from dgl.data import citation_graph as citegrh


@dataclass
class EdgeSplit:
    train_pos: torch.Tensor
    val_pos: torch.Tensor
    test_pos: torch.Tensor
    train_neg: torch.Tensor
    val_neg: torch.Tensor
    test_neg: torch.Tensor


class CoraLinkPredDataset:
    """Prepare a link prediction dataset on Cora.

    Protocol:
    1. Load full Cora graph + features (features+structure regime).
    2. Undirected edge list (keep only u < v to avoid duplicates).
    3. Random split: default 85/5/10 train/val/test for positive edges.
    4. For each split sample equal-number negative edges (non-edges).
    5. Provide simple query API: f_e(u,v) returns victim logits later.

    Attributes exposed align with existing attack base expectations:
    - graph_data (DGLGraph)
    - graph_dataset (raw dataset object from dgl)
    - num_nodes, num_features, num_classes (binary task => 2)
    - api_type = 'dgl'
    - edge_split: EdgeSplit dataclass with pos/neg tensors shape [2, N]
    """

    def __init__(self, train_ratio: float = 0.85, val_ratio: float = 0.05, seed: int = 42):
        assert train_ratio + val_ratio < 1.0, "Train + val must be < 1.0"
        self.api_type = "dgl"
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.graph_dataset = citegrh.load_cora()
        g_raw = self.graph_dataset[0]
        # Ensure undirected (Cora already is, but we symmetrize defensively) while preserving node data
        g = dgl.to_bidirected(g_raw)
        g = dgl.to_simple(g)
        for k, v in g_raw.ndata.items():  # restore features/masks/labels if lost
            if k not in g.ndata:
                g.ndata[k] = v
        self.graph_data = g

        self.num_nodes = g.num_nodes()
        self.num_features = g.ndata['feat'].shape[1]
        # Binary link existence classification
        self.num_classes = 2

        self.edge_split = self._build_edge_split(train_ratio, val_ratio)

    def _build_edge_split(self, train_ratio: float, val_ratio: float) -> EdgeSplit:
        # Get unique undirected edges
        src, dst = self.graph_data.edges()
        uv = torch.stack([src, dst], dim=0)
        mask = uv[0] < uv[1]
        uv = uv[:, mask]
        num_edges = uv.shape[1]

        # Shuffle indices
        perm = torch.randperm(num_edges)
        train_end = int(train_ratio * num_edges)
        val_end = train_end + int(val_ratio * num_edges)
        train_pos = uv[:, perm[:train_end]]
        val_pos = uv[:, perm[train_end:val_end]]
        test_pos = uv[:, perm[val_end:]]

        # Negative sampling helper
        def sample_neg(count: int) -> torch.Tensor:
            neg = []
            existing = set((int(u), int(v)) for u, v in zip(uv[0].tolist(), uv[1].tolist()))
            attempts = 0
            while len(neg) < count and attempts < count * 10:
                u = random.randrange(self.num_nodes)
                v = random.randrange(self.num_nodes)
                if u == v:
                    attempts += 1
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in existing:
                    attempts += 1
                    continue
                neg.append((a, b))
            if len(neg) < count:
                raise RuntimeError("Could not sample enough negative edges")
            return torch.tensor(neg, dtype=torch.long).t()

        train_neg = sample_neg(train_pos.shape[1])
        val_neg = sample_neg(val_pos.shape[1])
        test_neg = sample_neg(test_pos.shape[1])

        return EdgeSplit(train_pos, val_pos, test_pos, train_neg, val_neg, test_neg)

    def edge_batches(self, split: str, batch_size: int = 1024, shuffle: bool = True):
        if split == 'train':
            pos = self.edge_split.train_pos
            neg = self.edge_split.train_neg
        elif split == 'val':
            pos = self.edge_split.val_pos
            neg = self.edge_split.val_neg
        elif split == 'test':
            pos = self.edge_split.test_pos
            neg = self.edge_split.test_neg
        else:
            raise ValueError(f"Unknown split {split}")

        # Build edge tensor [2, N] and labels
        edges = torch.cat([pos, neg], dim=1)
        labels = torch.cat([
            torch.ones(pos.shape[1], dtype=torch.long),
            torch.zeros(neg.shape[1], dtype=torch.long)
        ])
        idx = torch.arange(labels.shape[0])
        if shuffle:
            perm = torch.randperm(idx.shape[0])
            idx = idx[perm]
        for i in range(0, idx.shape[0], batch_size):
            sel = idx[i:i + batch_size]
            yield edges[:, sel], labels[sel]
