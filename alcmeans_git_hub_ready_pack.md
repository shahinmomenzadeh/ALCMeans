# ALCMeans — GitHub-ready Pack

Below is a ready-to-paste repo layout with key files. Copy each section into the corresponding file.

---

## 📁 Repository structure
```
ALCMeans/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ data/
│  ├─ Football.csv            # your edges
│  └─ Football_gt.txt         # ground-truth (node label per line)
├─ src/
│  └─ alcmeans/
│     ├─ __init__.py
│     ├─ run.py               # main entry (CLI)
│     ├─ deepwalk.py
│     ├─ clustering.py
│     ├─ metrics.py
│     └─ utils.py
└─ tests/
   └─ test_smoke.py
```

---

## README.md
```markdown
# ALCMeans: Energy-based Seeding + DeepWalk + KMeans

This repo implements an energy-based community detection pipeline:

1. **Laplacian energy** per node: \(E(v)=d(v)^2 + d(v) + 2\sum_{u\in N(v)} d(u)\)
2. Seed formation: iterate nodes by descending energy; create cluster `{v} ∪ N(v)` if `v` not yet covered (centers unique; neighbors may overlap).
3. Merge clusters with non-commutative overlap similarity \(S_{max}\); repeat until convergence.
4. Pick the **center** in each merged cluster as the node with max energy.
5. **DeepWalk** embeddings → **KMeans** initialized with chosen centers.
6. Report **NMI**, **ARI**, **Modularity**, **F1-macro/micro**.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data format
- `Football.csv`: edge list with two columns: `n1,n2` (header required)
- `Football_gt.txt`: space-separated `node label` per line (no header)

## Usage
```bash
python -m alcmeans.run \
  --edges data/Football.csv \
  --ground-truth data/Football_gt.txt \
  --vector-size 7 --num-walks 90 --walk-length 40 \
  --similarity-threshold 0.5 --epochs 10 --negative 10 \
  --seed 42
```

Optional sweep (quick):
```bash
python -m alcmeans.run --edges data/Football.csv --ground-truth data/Football_gt.txt \
  --sweep --vector-sizes 5 7 10 --num-walks-list 50 70 90 --walk-lengths 20 40 60
```

## Notes
- Determinism is best-effort due to Word2Vec multithreading; we set seeds and workers=1 by default for reproducibility.
- `scipy` is optional but enables optimal label mapping (Hungarian).

## License
MIT
```

---

## requirements.txt
```txt
pandas>=2.0
numpy>=1.24
networkx>=3.1
scikit-learn>=1.3
gensim>=4.3
scipy>=1.11
```

---

## .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Data (keep a sample only)
*.csv
*.txt
!data/.gitkeep
```

---

## LICENSE (MIT)
```text
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
```

---

## src/alcmeans/__init__.py
```python
__all__ = [
    "laplacian_energy",
    "build_energy_based_clusters",
    "optimized_deepwalk",
    "evaluate_all_metrics",
]
```

---

## src/alcmeans/utils.py
```python
from __future__ import annotations
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_node_order_dict(nodes):
    return {n: i for i, n in enumerate(nodes)}
```

---

## src/alcmeans/deepwalk.py
```python
from __future__ import annotations
import numpy as np
import networkx as nx
from gensim.models import Word2Vec


def optimized_deepwalk(G: nx.Graph, vector_size: int, num_walks: int, walk_length: int,
                       restart_prob: float = 0.15, workers: int = 1, epochs: int = 10,
                       negative: int = 10, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    walks = []
    nodes = list(G.nodes())

    for _ in range(num_walks):
        for start_node in nodes:
            walk = [start_node]
            while len(walk) < walk_length:
                current = walk[-1]
                neighbors = list(G.neighbors(current))
                if not neighbors or rng.random() < restart_prob:
                    next_node = rng.choice(nodes)
                else:
                    if nx.is_weighted(G):
                        weights = np.array([G[current][nbr].get("weight", 1.0) for nbr in neighbors], dtype=float)
                        weights /= weights.sum()
                        next_node = rng.choice(neighbors, p=weights)
                    else:
                        next_node = rng.choice(neighbors)
                walk.append(next_node)
            walks.append(walk)

    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=10,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=epochs,
        negative=negative,
        seed=seed,
    )

    idx = {node: i for i, node in enumerate(nodes)}
    emb = np.zeros((len(nodes), vector_size), dtype=float)
    for node in nodes:
        emb[idx[node]] = model.wv[node]
    return emb
```

---

## src/alcmeans/clustering.py
```python
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Set


def laplacian_energy(G: nx.Graph) -> Dict:
    deg = dict(G.degree())
    out = {}
    for v in G.nodes():
        dv = deg[v]
        sum_nei = sum(deg[u] for u in G.neighbors(v))
        out[v] = dv * dv + dv + 2 * sum_nei
    return out


def smax_overlap(c1: Set, c2: Set) -> float:
    if not c1 or not c2:
        return 0.0
    inter = len(c1 & c2)
    if inter == 0:
        return 0.0
    return max(inter / len(c1), inter / len(c2))


def build_energy_based_clusters(G: nx.Graph, energies: Dict, similarity_threshold: float = 0.5) -> List[Set]:
    nodes_sorted = [n for n, e in sorted(energies.items(), key=lambda x: x[1], reverse=True)]
    clusters: List[Set] = []
    covered = set()

    for n in nodes_sorted:
        if n not in covered:
            c = set([n]) | set(G.neighbors(n))
            clusters.append(c)
            covered |= c
        if len(covered) == len(G):
            break

    if len(covered) < len(G):
        for r in (set(G.nodes()) - covered):
            clusters.append({r})
            covered.add(r)

    # first pass merge
    merged: List[Set] = []
    for c in clusters:
        placed = False
        for i, mc in enumerate(merged):
            if smax_overlap(c, mc) > similarity_threshold:
                merged[i] = mc | c
                placed = True
                break
        if not placed:
            merged.append(c)

    # repeated merges until convergence (bug-fixed: correct variable names)
    changed = True
    while changed:
        changed = False
        new_merged: List[Set] = []
        used = [False] * len(merged)
        for i, c1 in enumerate(merged):
            if used[i]:
                continue
            cur = set(c1)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                c2 = merged[j]
                if smax_overlap(cur, c2) > similarity_threshold:
                    cur |= c2
                    used[j] = True
                    changed = True
            new_merged.append(cur)
        merged = new_merged

    return merged
```

---

## src/alcmeans/metrics.py
```python
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, confusion_matrix

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from networkx.algorithms.community.quality import modularity as nx_modularity


def labels_to_communities(y_pred: np.ndarray, node_order: List) -> List[set]:
    comms = {}
    for n, c in zip(node_order, y_pred):
        comms.setdefault(int(c), set()).add(n)
    return list(comms.values())


def modularity_score(G: nx.Graph, y_pred: np.ndarray, node_order: List) -> float:
    return nx_modularity(G, labels_to_communities(y_pred, node_order))


def map_pred_to_true_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    if _HAS_SCIPY:
        r_ind, c_ind = linear_sum_assignment(-cm)
        mapping = {pred: true for true, pred in zip(r_ind, c_ind)}
        # cover any unmapped preds
        for p in set(np.unique(y_pred)) - set(mapping.keys()):
            mapping[p] = int(np.argmax(cm[:, p]))
    else:
        mapping = {p: int(np.argmax(cm[:, p])) for p in np.unique(y_pred)}
    return np.array([mapping[p] for p in y_pred], dtype=int)


def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_pm = map_pred_to_true_hungarian(y_true, y_pred)
    return (
        f1_score(y_true, y_pm, average="macro", zero_division=0),
        f1_score(y_true, y_pm, average="micro", zero_division=0),
    )


def evaluate_all_metrics(G: nx.Graph, nodes_list: List, ground_truth: Dict, labels: np.ndarray):
    gt = np.array([ground_truth.get(n, -1) for n in nodes_list])
    valid = gt != -1
    if valid.sum() == 0:
        print("No valid ground truth labels found for evaluation.")
        return None

    y_true = gt[valid]
    y_pred = np.asarray(labels)[valid]

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mod = modularity_score(G, np.asarray(labels), nodes_list)
    f1_mac, f1_mic = f1_macro_micro(y_true, y_pred)

    print(f"NMI: {nmi:.6f}")
    print(f"ARI: {ari:.6f}")
    print(f"Modularity: {mod:.6f}")
    print(f"F1-macro: {f1_mac:.6f}")
    print(f"F1-micro: {f1_mic:.6f}")

    return dict(NMI=nmi, ARI=ari, Modularity=mod, F1_macro=f1_mac, F1_micro=f1_mic)
```

---

## src/alcmeans/run.py (CLI)
```python
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans

from .clustering import laplacian_energy, build_energy_based_clusters
from .deepwalk import optimized_deepwalk
from .metrics import evaluate_all_metrics, modularity_score
from .utils import set_seed


def load_graph(edges_path: str) -> nx.Graph:
    df = pd.read_csv(edges_path)
    if not {"n1", "n2"}.issubset(df.columns):
        raise ValueError("Edge file must have columns 'n1' and 'n2'")
    return nx.from_pandas_edgelist(df, source="n1", target="n2")


def load_ground_truth(gt_path: str) -> dict:
    gt = pd.read_csv(gt_path, sep=" ", header=None, names=["node", "label"])  # space-separated
    return dict(zip(gt["node"], gt["label"]))


def run_once(G: nx.Graph, ground_truth: dict, vector_size: int, num_walks: int, walk_length: int,
             similarity_threshold: float, epochs: int, negative: int, seed: int):
    set_seed(seed)

    energies = laplacian_energy(G)
    merged_clusters = build_energy_based_clusters(G, energies, similarity_threshold)
    centers = [max(c, key=lambda n: energies.get(n, -np.inf)) for c in merged_clusters]

    print(f"Number of merged clusters = {len(merged_clusters)}")
    print(f"Centers after merging: {centers}")

    emb = optimized_deepwalk(G, vector_size, num_walks, walk_length,
                             workers=1, epochs=epochs, negative=negative, seed=seed)

    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    init_centroids = np.array([emb[node_to_idx[c]] for c in centers])

    if len(init_centroids) == 0:
        raise RuntimeError("No clusters found.")

    kmeans = KMeans(n_clusters=len(init_centroids), init=init_centroids, n_init=1, random_state=seed)
    labels = kmeans.fit_predict(emb)

    return evaluate_all_metrics(G, nodes_list, ground_truth, labels)


def run_sweep(G: nx.Graph, ground_truth: dict, vector_sizes, num_walks_list, walk_lengths,
              similarity_threshold: float, epochs: int, negative: int, seed: int):
    best_nmi, best_ari = -1.0, -1.0
    best_params = None
    for vs in vector_sizes:
        for nw in num_walks_list:
            for wl in walk_lengths:
                print(f"\nTesting vs={vs}, nw={nw}, wl={wl}")
                metrics = run_once(G, ground_truth, vs, nw, wl, similarity_threshold, epochs, negative, seed)
                if not metrics:
                    continue
                nmi, ari = metrics["NMI"], metrics["ARI"]
                if (nmi > best_nmi) or (ari > best_ari):
                    best_nmi, best_ari = nmi, ari
                    best_params = (vs, nw, wl)
    print(f"\nBest NMI: {best_nmi}, Best ARI: {best_ari} with params {best_params}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--vector-size", type=int, default=7)
    p.add_argument("--num-walks", type=int, default=90)
    p.add_argument("--walk-length", type=int, default=40)
    p.add_argument("--similarity-threshold", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--negative", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--vector-sizes", nargs="*", type=int)
    p.add_argument("--num-walks-list", nargs="*", type=int)
    p.add_argument("--walk-lengths", nargs="*", type=int)
    args = p.parse_args()

    G = load_graph(args.edges)
    gt = load_ground_truth(args.ground_truth)

    if args.sweep:
        vs = args.vector_sizes or [5, 7, 10]
        nw = args.num_walks_list or [50, 70, 90]
        wl = args.walk_lengths or [20, 40, 60]
        run_sweep(G, gt, vs, nw, wl, args.similarity_threshold, args.epochs, args.negative, args.seed)
    else:
        run_once(G, gt, args.vector_size, args.num_walks, args.walk_length, args.similarity_threshold, args.epochs, args.negative, args.seed)


if __name__ == "__main__":
    main()
```

---

## tests/test_smoke.py
```python
import os

def test_repo_exists():
    assert os.path.exists("src/alcmeans/run.py")
```

---
