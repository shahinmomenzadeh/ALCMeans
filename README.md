ALCMean's: Unsupervised Community Detection Using Local Laplacian and Automatic Center Detection
ALCMean's (Automatic Laplacian Centrality Means) is an unsupervised community detection algorithm for complex networks. The method automatically identifies structurally important cluster centers using local Laplacian energy, merges overlapping seed communities, and refines final assignments using DeepWalk embeddings and K-Means initialized with the selected centers.
This repository contains the implementation and experimental workflow for the paper:
> S. Momenzadeh and R. Pir Mohammadiani, **"ALCMean's: Unsupervised Community Detection Using Local Laplacian, Automatic Detection of the Number of Centers"**, *Journal of Mahani Mathematical Research*, 15(2), 269-288, 2026. DOI: `10.22103/jmmr.2026.25756.1849`
---
Why ALCMean's?
Many community detection and clustering methods require the number of communities to be known in advance or are sensitive to random initialization. ALCMean's addresses these issues by combining graph-theoretic center discovery with embedding-based refinement:
It does not require a predefined number of communities.
It selects initial centers using local Laplacian energy, rather than random initialization.
It uses overlap-based merging to reduce redundant seed communities.
It applies DeepWalk embeddings to capture higher-order graph structure.
It performs final clustering using K-Means initialized by structurally selected centers.
---
Method Overview
The algorithm follows four main stages.
1. Laplacian Energy-Based Node Importance
For each node `v`, ALCMean's computes a local Laplacian energy score:
```text
E(v) = d(v)^2 + d(v) + 2 * Σ d(u),  u ∈ N(v)
```
where `d(v)` is the degree of node `v` and `N(v)` is its open neighborhood. Nodes with higher energy are treated as more structurally influential.
2. Seed Community Formation
Nodes are sorted in descending order of energy. Each uncovered high-energy node creates an initial seed community containing itself and its immediate neighbors:
```text
C = {v} ∪ N(v)
```
This creates local communities around structurally important nodes.
3. Similarity-Based Community Merging
Initial seed communities may overlap. ALCMean's merges communities using an asymmetric overlap score:
```text
Smax(Ci, Cj) = max(|Ci ∩ Cj| / |Ci|, |Ci ∩ Cj| / |Cj|)
```
If `Smax ≥ 0.5`, the two communities are merged. This step helps avoid fragmented and redundant clusters.
4. DeepWalk Embedding and Final Assignment
After merging, the node with the highest Laplacian energy inside each community is selected as the final center. DeepWalk is then used to generate node embeddings, and K-Means is initialized with the embeddings of the selected centers. Each node is assigned to the nearest center in the embedding space.
---
Algorithm Summary
```text
Input: Graph G=(V,E), vector size, number of walks, walk length
Output: Detected communities

1. Compute Laplacian energy E(v) for all nodes.
2. Sort nodes by E(v) in descending order.
3. Create seed clusters from uncovered high-energy nodes and their neighbors.
4. Merge overlapping clusters while Smax ≥ 0.5.
5. Select one final center per merged cluster using maximum E(v).
6. Generate DeepWalk embeddings.
7. Run K-Means initialized with the selected centers.
8. Assign each node to the nearest center.
```
---
Main Features
Unsupervised community detection
Automatic detection of the number of centers/communities
Local Laplacian energy-based center selection
Overlap-based seed community merging
DeepWalk-based node representation learning
K-Means with deterministic structural initialization
Evaluation using NMI, ARI, modularity, F1-macro, and F1-micro
Suitable for small and medium-sized benchmark networks
---
Benchmark Datasets
The paper evaluates ALCMean's on five benchmark datasets:
Dataset	Nodes	Edges	Ground-truth Communities
Football	115	611	12
Texas	187	298	5
Cornell	183	295	5
Washington	230	446	5
email-Eu-core	1005	25571	42
---
Reported Results
The published experiments compare ALCMean's with Louvain, Newman-Girvan, Label Propagation, Fast-Greedy, and MAGI (KDD 2024). The proposed method achieved strong results across the evaluated datasets, especially on NMI and ARI.
Dataset	NMI	ARI	Modularity	F1-macro	F1-micro
Football	0.91	0.89	0.55	0.72	0.87
Texas	0.19	0.07	0.459	0.03	0.232
Cornell	0.24	0.03	0.53	0.02	0.20
Washington	0.184	0.30	0.47	0.03	0.20
email-Eu-core	0.70	0.39	0.22	0.19	0.48
The ablation study reported in the paper shows that removing DeepWalk produces the largest performance drop, while removing Laplacian energy or the merging step also reduces stability and accuracy.
---
Recommended Repository Structure
For a clean public GitHub repository, the project should be organized as real source files rather than keeping the implementation inside a single markdown file.
```text
ALCMeans/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   ├── Football.csv
│   ├── Football_gt.txt
│   └── ...
├── src/
│   └── alcmeans/
│       ├── __init__.py
│       ├── run.py
│       ├── deepwalk.py
│       ├── clustering.py
│       ├── metrics.py
│       └── utils.py
├── experiments/
│   ├── run_all.py
│   └── configs/
└── tests/
    └── test_smoke.py
```
---
Installation
Create and activate a virtual environment:
```bash
python -m venv .venv
```
On Windows:
```bash
.venv\Scripts\activate
```
On Linux/macOS:
```bash
source .venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
---
Requirements
A typical Python environment for this project includes:
```text
python>=3.9
numpy>=1.24
scipy>=1.10
pandas>=2.0
networkx>=3.0
scikit-learn>=1.3
gensim>=4.3
matplotlib>=3.7
seaborn>=0.12
```
---
Usage Example
Run ALCMean's on a graph dataset:
```bash
python -m alcmeans.run \
  --edges data/Football.csv \
  --ground-truth data/Football_gt.txt \
  --vector-size 7 \
  --num-walks 90 \
  --walk-length 60 \
  --similarity-threshold 0.5 \
  --seed 42
```
Run experiments on all benchmark datasets:
```bash
python experiments/run_all.py
```
---
Data Format
Edge List
The edge file should contain graph edges, for example:
```text
source,target
0,1
1,2
2,3
```
Ground Truth Labels
The ground-truth file should contain one node and one label per line:
```text
0 1
1 1
2 2
3 2
```
---
Evaluation Metrics
The implementation can report the following metrics:
Normalized Mutual Information (NMI)
Adjusted Rand Index (ARI)
Modularity (Q)
F1-macro
F1-micro
Runtime
---
Limitations
ALCMean's is a research-oriented algorithm and has some practical limitations:
DeepWalk adds computational cost compared with lightweight heuristic methods.
The final results may depend on embedding parameters such as vector size, number of walks, and walk length.
The Laplacian energy score is mainly degree-based and may not fully capture complex multi-scale dependencies.
The current version is best suited for small and medium-sized graph datasets.
Statistical significance testing and larger-scale experiments can be added in future versions.
---
Citation
If you use this code or build on this work, please cite:
```bibtex
@article{momenzadeh2026alcmeans,
  title   = {ALCMean's: Unsupervised Community Detection Using Local Laplacian, Automatic Detection of the Number of Centers},
  author  = {Momenzadeh, Shahin and Pir Mohammadiani, Rojiar},
  journal = {Journal of Mahani Mathematical Research},
  volume  = {15},
  number  = {2},
  pages   = {269--288},
  year    = {2026},
  doi     = {10.22103/jmmr.2026.25756.1849}
}
```

Project Status
This repository is a research implementation of the ALCMean's community detection algorithm. It is suitable for demonstrating work in graph mining, unsupervised learning, network analysis, representation learning, and community detection.
Click on alcmeans_git_hub_ready_pack.md. The code and all the explanations are in the file.
