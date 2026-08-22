"""
Step 7: Graph Construction

Graph A: Bipartite graph
  Left nodes  : subjects (asignaturas)
  Right nodes : keywords shared by ≥ MIN_SUBJECTS_FOR_SHARED_KEYWORD subjects
  Edges       : subject uses that keyword

Graph B: Subject-to-subject graph
  Nodes : subjects
  Edges : two subjects share ≥ MIN_SHARED_KEYWORDS_FOR_EDGE keywords
  Weight: number of shared keywords (→ edge thickness)
"""

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

matplotlib.use("Agg")

from config import (
    MIN_SHARED_KEYWORDS_FOR_EDGE,
    MIN_SUBJECTS_FOR_SHARED_KEYWORD,
    OUTPUTS_DIR,
    TOP_N_KEYWORDS,
)

GRAPH_A_PATH = OUTPUTS_DIR / "07_graph_bipartite.png"
GRAPH_B_PATH = OUTPUTS_DIR / "07_graph_subjects.png"


def _load_counters() -> dict[str, Counter]:
    df = pd.read_csv(OUTPUTS_DIR / "06_keywords.csv")
    counters: dict[str, Counter] = {}
    for _, row in df.iterrows():
        counters.setdefault(row["subject"], Counter())[row["keyword"]] = row["frequency"]
    return counters


def _build_subject_vocab(counters: dict[str, Counter], top_n: int = TOP_N_KEYWORDS) -> dict[str, set]:
    return {subj: {w for w, _ in c.most_common(top_n)} for subj, c in counters.items()}


def _find_shared_keywords(vocab: dict[str, set]) -> Counter:
    """Count in how many subjects each keyword appears."""
    global_count: Counter = Counter()
    for keywords in vocab.values():
        for kw in keywords:
            global_count[kw] += 1
    return global_count


def build_bipartite_graph(vocab: dict[str, set], global_count: Counter) -> nx.Graph:
    G = nx.Graph()
    subjects = sorted(vocab.keys())
    shared_kws = {kw for kw, n in global_count.items() if n >= MIN_SUBJECTS_FOR_SHARED_KEYWORD}

    for subj in subjects:
        G.add_node(subj, bipartite=0)
    for kw in shared_kws:
        G.add_node(kw, bipartite=1)
    for subj, keywords in vocab.items():
        for kw in keywords & shared_kws:
            G.add_edge(subj, kw)
    return G


def plot_bipartite(G: nx.Graph, vocab: dict[str, set]) -> None:
    subjects = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    keywords = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    pos = {}
    for i, s in enumerate(sorted(subjects)):
        pos[s] = (-1, i * 2)
    for i, k in enumerate(sorted(keywords)):
        pos[k] = (1, i * (len(subjects) * 2 / max(len(keywords), 1)))

    fig, ax = plt.subplots(figsize=(14, max(10, len(keywords) * 0.4)))

    nx.draw_networkx_nodes(G, pos, nodelist=subjects, node_color="#4C72B0",
                           node_size=1800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=keywords, node_color="#DD8452",
                           node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in subjects},
                            font_size=9, font_color="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in keywords},
                            font_size=7, ax=ax)

    ax.set_title("Grafo bipartito: Asignaturas ↔ Palabras clave compartidas", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(GRAPH_A_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bipartite graph → {GRAPH_A_PATH}")


def build_subject_graph(vocab: dict[str, set]) -> nx.Graph:
    subjects = sorted(vocab.keys())
    G = nx.Graph()
    G.add_nodes_from(subjects)
    for i, s1 in enumerate(subjects):
        for s2 in subjects[i + 1:]:
            shared = vocab[s1] & vocab[s2]
            if len(shared) >= MIN_SHARED_KEYWORDS_FOR_EDGE:
                G.add_edge(s1, s2, weight=len(shared), shared=shared)
    return G


def plot_subject_graph(G: nx.Graph) -> None:
    if G.number_of_edges() == 0:
        print("No edges in subject graph (try lowering MIN_SHARED_KEYWORDS_FOR_EDGE).")
        return

    pos = nx.spring_layout(G, seed=42, k=2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [1 + 4 * (w / max_w) for w in weights]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="#4C72B0", alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="black", ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, edge_color="#888", ax=ax)

    edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(
        "Grafo de asignaturas: ejes = palabras clave compartidas\n"
        f"(grosor ∝ cantidad compartida, mínimo {MIN_SHARED_KEYWORDS_FOR_EDGE})",
        fontsize=13,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(GRAPH_B_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subject graph → {GRAPH_B_PATH}")


def print_analysis(G: nx.Graph, vocab: dict[str, set]) -> None:
    print("=" * 60)
    print("STEP 7: GRAPH ANALYSIS")
    print("=" * 60)
    print(f"Subject nodes : {G.number_of_nodes()}")
    print(f"Edges (shared): {G.number_of_edges()}")
    print()
    print("Shared keywords per pair:")
    for u, v, data in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
        shared_list = ", ".join(sorted(data["shared"])[:8])
        print(f"  {u} ↔ {v}  ({data['weight']} kws): {shared_list}")
    print()
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        most_connected = max(degrees, key=degrees.get)
        print(f"Most connected subject: {most_connected} (degree {degrees[most_connected]})")


def run(counters: dict[str, Counter] = None) -> None:
    if counters is None:
        counters = _load_counters()

    vocab = _build_subject_vocab(counters)
    global_count = _find_shared_keywords(vocab)

    G_bip = build_bipartite_graph(vocab, global_count)
    plot_bipartite(G_bip, vocab)

    G_subj = build_subject_graph(vocab)
    print_analysis(G_subj, vocab)
    plot_subject_graph(G_subj)


if __name__ == "__main__":
    run()
