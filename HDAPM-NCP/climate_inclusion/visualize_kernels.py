"""
Visualize herb and disease kernels as graphs.

Creates KNN graphs from similarity kernels and renders PNGs using a spring layout.
Also exports GEXF files for interactive exploration in Gephi/NetworkX.

Default inputs:
- Herb (filtered baseline): data/herb_kernel/climate_filtered/herb_target.txt
- Herb (climate-only):      data/herb_kernel/climate_filtered/herb_climate_kernel.txt
- Disease:                  data/disease_kernel/disease_target.txt

Outputs (by default): data/herb_kernel/climate_filtered/graphs/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HERB_DIR = DATA_DIR / "herb_kernel"
FILTER_DIR = HERB_DIR / "climate_filtered"
DISEASE_DIR = DATA_DIR / "disease_kernel"
OUT_DIR = FILTER_DIR / "graphs"


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.loadtxt(path, dtype=float, delimiter=',')


def load_labels_herb_filtered() -> list[str]:
    # file format: index,HerbId
    p = FILTER_DIR / 'herb_id.filtered.csv'
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p, header=None, names=['idx', 'HerbId'])
    return df['HerbId'].tolist()


def load_labels_disease() -> list[str]:
    p = DISEASE_DIR / 'disease_id.csv'
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p, header=None, names=['idx', 'DiseaseId'])
    return df['DiseaseId'].tolist()


def knn_edges_from_kernel(K: np.ndarray, k: int, min_weight: float = 0.0):
    """Return undirected edge list (i,j,weight) using symmetric KNN on similarity matrix."""
    n = K.shape[0]
    # zero diagonal to avoid self loops
    K = K.copy()
    np.fill_diagonal(K, 0.0)
    knn_sets = []
    for i in range(n):
        idx = np.argsort(-K[i])[:k]
        knn_sets.append(set(int(j) for j in idx if K[i, j] >= min_weight))
    edges = {}
    for i in range(n):
        for j in knn_sets[i]:
            if i == j:
                continue
            if (j in knn_sets[i]) or (i in knn_sets[j]):
                w = float(max(K[i, j], K[j, i]))
                key = (min(i, j), max(i, j))
                if key not in edges or w > edges[key]:
                    edges[key] = w
    return [(i, j, w) for (i, j), w in edges.items()]


def build_graph(K: np.ndarray, labels: list[str], k: int = 5, min_weight: float = 0.0) -> nx.Graph:
    G = nx.Graph()
    for i, lab in enumerate(labels):
        G.add_node(i, label=str(lab))
    for i, j, w in knn_edges_from_kernel(K, k=k, min_weight=min_weight):
        G.add_edge(i, j, weight=w)
    return G


def draw_graph(G: nx.Graph, out_path: Path, title: str = "Graph", node_size_scale: float = 800.0, show_labels: bool = False):
    if not HAVE_PLOT:
        print("Matplotlib not available; skipping plot for", out_path)
        return
    pos = nx.spring_layout(G, weight='weight', seed=42)
    deg = dict(G.degree())
    sizes = np.array([deg[n] for n in G.nodes()], dtype=float)
    if sizes.size > 0:
        sizes = (sizes - sizes.min()) / (sizes.ptp() + 1e-9) * node_size_scale + 100
    else:
        sizes = 200
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    # normalize edge width
    if len(weights) > 0:
        ws = np.array(weights)
        ws = 0.5 + 3.0 * (ws - ws.min()) / (ws.ptp() + 1e-9)
    else:
        ws = []
    plt.figure(figsize=(8, 7))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#1f77b4", alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=ws, alpha=0.5)
    if show_labels:
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8)
    plt.title(title)
    plt.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def export_gexf(G: nx.Graph, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, out_path)


def _add_distance_attr(G: nx.Graph) -> nx.Graph:
    """Return a copy with 'distance' attribute as inverse of 'weight'."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = float(data.get('weight', 1.0))
        dist = 1.0 / max(w, 1e-12)
        H.add_edge(u, v, **data)
        H[u][v]['distance'] = dist
    return H


def compute_node_metrics(G: nx.Graph, labels: list[str]) -> pd.DataFrame:
    # Degrees
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight='weight'))

    # Centralities
    betw = nx.betweenness_centrality(G, weight='weight', normalized=True)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception:
        eig = {n: np.nan for n in G.nodes()}
    pr = nx.pagerank(G, weight='weight', alpha=0.85)

    # Closeness uses a distance attribute (inverse similarity)
    Gd = _add_distance_attr(G)
    close = nx.closeness_centrality(Gd, distance='distance')

    # Clustering
    clust = nx.clustering(G, weight='weight')

    # Core numbers (unweighted)
    try:
        core = nx.core_number(G)
    except Exception:
        core = {n: 0 for n in G.nodes()}

    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            'node': i,
            'label': lab,
            'degree': deg.get(i, 0),
            'weighted_degree': float(wdeg.get(i, 0.0)),
            'betweenness': float(betw.get(i, 0.0)),
            'closeness': float(close.get(i, 0.0)),
            'eigenvector': float(eig.get(i, np.nan)),
            'pagerank': float(pr.get(i, 0.0)),
            'clustering_coef': float(clust.get(i, 0.0)),
            'core_number': int(core.get(i, 0)),
        })
    return pd.DataFrame(rows)


def _weighted_diameter_LCC(G: nx.Graph) -> float | None:
    comps = list(nx.connected_components(G))
    if not comps:
        return None
    lcc_nodes = max(comps, key=len)
    if len(lcc_nodes) < 2:
        return None
    Glcc = G.subgraph(lcc_nodes).copy()
    Glcc_d = _add_distance_attr(Glcc)
    # Max of all-pairs shortest path distances (Dijkstra) on LCC
    maxdist = 0.0
    try:
        for _, lengths in nx.all_pairs_dijkstra_path_length(Glcc_d, weight='distance'):
            if lengths:
                md = max(lengths.values())
                if md > maxdist:
                    maxdist = md
        return float(maxdist)
    except Exception:
        return None


def compute_graph_summary(G: nx.Graph, labels: list[str]) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_clust = nx.average_clustering(G, weight='weight') if n > 0 else 0.0
    trans = nx.transitivity(G) if n > 0 else 0.0
    comps = list(nx.connected_components(G))
    num_comp = len(comps)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    lcc_n = comp_sizes[0] if comp_sizes else 0

    # Average shortest path on largest component using distance (inverse weight)
    asp = None
    if lcc_n >= 2:
        lcc_nodes = max(comps, key=len)
        Glcc = G.subgraph(lcc_nodes).copy()
        Glcc_d = _add_distance_attr(Glcc)
        try:
            asp = nx.average_shortest_path_length(Glcc_d, weight='distance')
        except Exception:
            asp = None

    # Degree assortativity (weighted)
    try:
        assort = nx.degree_assortativity_coefficient(G, weight='weight')
    except Exception:
        assort = None

    # Communities and modularity
    comms = []
    mod = None
    try:
        communities = greedy_modularity_communities(G, weight='weight')
        comms = [sorted([labels[i] for i in comm]) for comm in communities]
        mod = modularity(G, communities, weight='weight')
    except Exception:
        pass

    # Degree statistics
    deg_arr = np.array([d for _, d in G.degree()], dtype=float) if n > 0 else np.array([])
    wdeg_arr = np.array([d for _, d in G.degree(weight='weight')], dtype=float) if n > 0 else np.array([])

    # Weighted diameter on LCC
    diam_w = _weighted_diameter_LCC(G)

    return {
        'nodes': n,
        'edges': m,
        'density': float(density),
        'average_clustering': float(avg_clust),
        'transitivity': float(trans),
        'num_components': num_comp,
        'component_sizes': comp_sizes,
        'largest_component_nodes': int(lcc_n),
        'avg_shortest_path_length_LCC': (float(asp) if asp is not None else None),
        'weighted_diameter_LCC': (float(diam_w) if diam_w is not None else None),
        'degree_assortativity_weighted': (float(assort) if assort is not None else None),
        'modularity_greedy_weighted': (float(mod) if mod is not None else None),
        'communities_labels': comms,
        'avg_degree': (float(deg_arr.mean()) if deg_arr.size else 0.0),
        'std_degree': (float(deg_arr.std()) if deg_arr.size else 0.0),
        'avg_weighted_degree': (float(wdeg_arr.mean()) if wdeg_arr.size else 0.0),
        'std_weighted_degree': (float(wdeg_arr.std()) if wdeg_arr.size else 0.0),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR = OUT_DIR / 'metrics'
    MET_DIR.mkdir(parents=True, exist_ok=True)

    # Herb (baseline filtered)
    herb_labels = load_labels_herb_filtered()
    herb_base_path = FILTER_DIR / 'herb_target.txt'
    herb_clim_path = FILTER_DIR / 'herb_climate_kernel.txt'
    if herb_base_path.exists():
        Kb = load_matrix(herb_base_path)
        Gh = build_graph(Kb, herb_labels, k=5, min_weight=0.0)
        draw_graph(Gh, OUT_DIR / 'herb_baseline_knn5.png', title='Herb Baseline KNN-5')
        export_gexf(Gh, OUT_DIR / 'herb_baseline_knn5.gexf')
        # Metrics
        df_h = compute_node_metrics(Gh, herb_labels)
        df_h.to_csv(MET_DIR / 'herb_baseline_node_metrics.csv', index=False)
        herb_base_summary = compute_graph_summary(Gh, herb_labels)
    else:
        print("[WARN] Missing:", herb_base_path)
        df_h = None
        herb_base_summary = None

    # Herb (climate-only)
    if herb_clim_path.exists():
        Kc = load_matrix(herb_clim_path)
        Ghc = build_graph(Kc, herb_labels, k=5, min_weight=0.0)
        draw_graph(Ghc, OUT_DIR / 'herb_climate_knn5.png', title='Herb Climate KNN-5')
        export_gexf(Ghc, OUT_DIR / 'herb_climate_knn5.gexf')
        df_hc = compute_node_metrics(Ghc, herb_labels)
        df_hc.to_csv(MET_DIR / 'herb_climate_node_metrics.csv', index=False)
        herb_clim_summary = compute_graph_summary(Ghc, herb_labels)
    else:
        print("[WARN] Missing:", herb_clim_path)
        df_hc = None
        herb_clim_summary = None

    # Disease
    dis_labels = load_labels_disease()
    dis_path = DISEASE_DIR / 'disease_target.txt'
    if dis_path.exists():
        Kd = load_matrix(dis_path)
        Gd = build_graph(Kd, dis_labels, k=5, min_weight=0.0)
        draw_graph(Gd, OUT_DIR / 'disease_knn5.png', title='Disease KNN-5')
        export_gexf(Gd, OUT_DIR / 'disease_knn5.gexf')
        # Disease metrics (could be large; still computed here)
        df_d = compute_node_metrics(Gd, dis_labels)
        df_d.to_csv(MET_DIR / 'disease_node_metrics.csv', index=False)
        disease_summary = compute_graph_summary(Gd, dis_labels)
    else:
        print("[WARN] Missing:", dis_path)
        df_d = None
        disease_summary = None

    # Save a summary JSON aggregating graph-level metrics
    summary = {
        'herb_baseline': herb_base_summary,
        'herb_climate': herb_clim_summary,
        'disease': disease_summary,
        'node_metrics_files': {
            'herb_baseline': str(MET_DIR / 'herb_baseline_node_metrics.csv') if df_h is not None else None,
            'herb_climate': str(MET_DIR / 'herb_climate_node_metrics.csv') if df_hc is not None else None,
            'disease': str(MET_DIR / 'disease_node_metrics.csv') if df_d is not None else None,
        }
    }
    with open(MET_DIR / 'graphs_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Also save a concise CSV of overall metrics (one row per graph)
    rows = []
    for name, summ in (
        ('herb_baseline', herb_base_summary),
        ('herb_climate', herb_clim_summary),
        ('disease', disease_summary),
    ):
        if summ is None:
            continue
        row = {'graph': name}
        # Select key numeric metrics for CSV
        for key in (
            'nodes','edges','density','average_clustering','transitivity','num_components',
            'largest_component_nodes','avg_shortest_path_length_LCC','weighted_diameter_LCC',
            'degree_assortativity_weighted','modularity_greedy_weighted','avg_degree','std_degree',
            'avg_weighted_degree','std_weighted_degree'
        ):
            row[key] = summ.get(key)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(MET_DIR / 'graphs_summary.csv', index=False)

    print("Saved graphs to:", OUT_DIR)


if __name__ == '__main__':
    main()
