#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter15/Twitter16 tree-time Hawkes-style rumor detection

What this script does
---------------------
1. Reads a Twitter15/Twitter16 dataset folder with:
   - label.txt
   - source_tweets.txt
   - tree/   (many *.txt tree files)
2. Extracts:
   - source tweet TF-IDF text features
   - propagation tree structure features
   - relative-time / Hawkes-style temporal features from each tree file
3. Trains baseline and full models using stratified train/dev/test split.
4. Prints evaluation and an experiment summary.

Expected dataset layout
-----------------------
<dataset_dir>/
    label.txt
    source_tweets.txt
    tree/
        80080680482123777.txt
        80084555733803009.txt
        ...

Important note about time
-------------------------
The tree file format released with the ACL'17 dataset stores relative posting delay
in MINUTES for each node as the 3rd field of a node tuple:
    ['uid', 'tweet_id', 'post_time_delay_in_minutes']
This is enough to build continuous-time / Hawkes-style features.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # feature extraction
    tfidf_max_features: int = 6000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_sublinear_tf: bool = True

    hawkes_decay: float = 0.08  # per minute; kernel exp(-beta * dt)
    hawkes_topk: int = 256      # use earliest top-k times for sequence features

    # split
    test_size: float = 0.25
    dev_size_from_train: float = 0.15
    random_state: int = 42

    # classifier
    clf_C: float = 3.0
    clf_solver: str = "liblinear"
    clf_max_iter: int = 2000

    # task
    task_mode: str = "binary"  # binary or four_class
    include_unverified_in_rumor: bool = True


# -----------------------------
# Parsing helpers
# -----------------------------
TREE_LINE_RE = re.compile(r"\s*(\[.*?\])\s*->\s*(\[.*\])\s*$")


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Could not read file: {path}")


def parse_label_file(path: Path, task_mode: str, include_unverified_in_rumor: bool) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for raw in safe_read_text(path).splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        label, event_id = line.split(":", 1)
        label = label.strip()
        event_id = event_id.strip()
        if task_mode == "binary":
            if label == "non-rumor":
                y = "non-rumor"
            elif label in {"false", "true"}:
                y = "rumor"
            elif label == "unverified":
                y = "rumor" if include_unverified_in_rumor else "non-rumor"
            else:
                continue
        elif task_mode == "four_class":
            y = label
        else:
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        label_map[event_id] = y
    return label_map


def parse_source_tweets(path: Path) -> Dict[str, str]:
    source = {}
    for raw in safe_read_text(path).splitlines():
        line = raw.rstrip("\n")
        if not line or "\t" not in line:
            continue
        event_id, text = line.split("\t", 1)
        source[event_id.strip()] = text.strip()
    return source


def parse_tree_node(node_str: str) -> Tuple[str, str, float]:
    vals = ast.literal_eval(node_str)
    if not isinstance(vals, list) or len(vals) != 3:
        raise ValueError(f"Bad node tuple: {node_str}")
    uid = str(vals[0])
    tweet_id = str(vals[1])
    try:
        delay = float(vals[2])
    except Exception:
        delay = 0.0
    return uid, tweet_id, delay


@dataclass
class TreeData:
    event_id: str
    root_uid: str
    root_delay: float
    node_delays: List[float]
    edges: List[Tuple[str, str]]
    children: Dict[str, List[str]]
    parent: Dict[str, Optional[str]]
    node_delay_by_uid: Dict[str, float]


def parse_tree_file(path: Path) -> TreeData:
    edges: List[Tuple[str, str]] = []
    children: Dict[str, List[str]] = defaultdict(list)
    parent: Dict[str, Optional[str]] = {}
    node_delay_by_uid: Dict[str, float] = {}
    root_uid = ""
    root_delay = 0.0
    event_id = path.stem

    lines = [ln for ln in safe_read_text(path).splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty tree file: {path}")

    for line in lines:
        m = TREE_LINE_RE.match(line)
        if not m:
            print(f"[DEBUG] Line doesn't match TREE_LINE_RE regex: {line}")
            continue
        left, right = m.group(1), m.group(2)


        l_uid, l_tid, l_delay = parse_tree_node(left)
        r_uid, r_tid, r_delay = parse_tree_node(right)
        if l_uid == "ROOT":
            root_uid = r_uid
            root_delay = r_delay
            parent[r_uid] = None
            node_delay_by_uid[r_uid] = r_delay
            event_id = r_tid if r_tid != "ROOT" else event_id
            continue
        edges.append((l_uid, r_uid))
        children[l_uid].append(r_uid)
        if l_uid not in parent:
            parent[l_uid] = None  # may get corrected later
        parent[r_uid] = l_uid
        node_delay_by_uid.setdefault(l_uid, l_delay)
        node_delay_by_uid[r_uid] = r_delay
        if not root_uid:
            root_uid = l_uid
            root_delay = l_delay

    # If ROOT line missing in a malformed file, infer root as node with no parent.
    if not root_uid and parent:
        roots = [u for u, p in parent.items() if p is None]
        if roots:
            root_uid = roots[0]
            root_delay = node_delay_by_uid.get(root_uid, 0.0)

    node_delays = sorted(max(0.0, d) for d in node_delay_by_uid.values())
    return TreeData(
        event_id=event_id,
        root_uid=root_uid,
        root_delay=root_delay,
        node_delays=node_delays,
        edges=edges,
        children=dict(children),
        parent=parent,
        node_delay_by_uid=node_delay_by_uid,
    )


# -----------------------------
# Feature extraction
# -----------------------------
def bfs_depth(children: Dict[str, List[str]], root: str) -> Tuple[int, float]:
    if not root:
        return 0, 0.0

    q = deque([(root, 0)])
    visited = set([root])
    depth_sum = 0
    cnt = 0
    max_depth = 0

    while q:
        node, dep = q.popleft()
        cnt += 1
        depth_sum += dep
        max_depth = max(max_depth, dep)

        for ch in children.get(node, []):
            if ch not in visited:
                visited.add(ch)
                q.append((ch, dep + 1))

    avg_depth = depth_sum / max(cnt, 1)
    return max_depth, avg_depth

def tree_structural_features(tree: TreeData) -> np.ndarray:
    num_nodes = len(tree.node_delay_by_uid)
    num_edges = len(tree.edges)
    out_degree = [len(v) for v in tree.children.values()]
    root_children = len(tree.children.get(tree.root_uid, [])) if tree.root_uid else 0
    max_depth, avg_depth = bfs_depth(tree.children, tree.root_uid)

    leaf_count = sum(1 for n in tree.node_delay_by_uid if len(tree.children.get(n, [])) == 0)
    max_out = max(out_degree) if out_degree else 0
    avg_out = float(np.mean(out_degree)) if out_degree else 0.0
    breadth_ratio = root_children / max(num_nodes - 1, 1)
    density_like = num_edges / max(num_nodes, 1)

    return np.array([
        num_nodes,
        num_edges,
        root_children,
        leaf_count,
        max_out,
        avg_out,
        max_depth,
        avg_depth,
        breadth_ratio,
        density_like,
    ], dtype=np.float32)


def hawkes_temporal_features(tree: TreeData, beta: float, topk: int) -> np.ndarray:
    times = np.array(sorted(max(0.0, t) for t in tree.node_delays), dtype=np.float64)
    if len(times) == 0:
        return np.zeros(31, dtype=np.float32)

    # unique/valid times after root. keep root at 0 if present.
    times = times[: max(1, topk)]
    post_times = times[times > 0]

    max_t = float(times.max())
    mean_t = float(times.mean())
    std_t = float(times.std())
    median_t = float(np.median(times))
    q25 = float(np.quantile(times, 0.25))
    q75 = float(np.quantile(times, 0.75))

    bins = [5, 10, 30, 60, 180, 360, 720, 1440]  # minutes
    bin_counts = [float(np.sum(times <= b)) for b in bins]
    bin_ratios = [c / max(len(times), 1) for c in bin_counts]

    # Hawkes-style cumulative excitation along ordered times
    if len(post_times) == 0:
        hawkes_sum = 0.0
        hawkes_max = 0.0
        hawkes_mean = 0.0
        burstiness = 0.0
        gap_mean = 0.0
        gap_std = 0.0
        gap_min = 0.0
        gap_max = 0.0
    else:
        excitation_values = []
        for i, t in enumerate(post_times):
            prev = post_times[:i]
            if len(prev) == 0:
                excitation_values.append(0.0)
            else:
                excitation_values.append(float(np.sum(np.exp(-beta * (t - prev)))))
        excitation = np.array(excitation_values, dtype=np.float64)
        hawkes_sum = float(excitation.sum())
        hawkes_max = float(excitation.max())
        hawkes_mean = float(excitation.mean())

        gaps = np.diff(np.concatenate(([0.0], post_times)))
        gap_mean = float(gaps.mean())
        gap_std = float(gaps.std())
        gap_min = float(gaps.min())
        gap_max = float(gaps.max())
        if gap_mean + gap_std == 0:
            burstiness = 0.0
        else:
            burstiness = float((gap_std - gap_mean) / (gap_std + gap_mean + 1e-8))

    feat = np.array([
        len(times),
        max_t,
        mean_t,
        std_t,
        median_t,
        q25,
        q75,
        *bin_counts,
        *bin_ratios,
        hawkes_sum,
        hawkes_max,
        hawkes_mean,
        gap_mean,
        gap_std,
        gap_min,
        gap_max,
        burstiness,
    ], dtype=np.float32)
    return feat


def build_dataset(
    dataset_dir: Path,
    cfg: Config,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    label_path = dataset_dir / "label.txt"
    source_path = dataset_dir / "source_tweets.txt"
    tree_dir = dataset_dir / "tree"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}")
    if not tree_dir.exists():
        raise FileNotFoundError(f"Missing {tree_dir}")

    labels = parse_label_file(label_path, cfg.task_mode, cfg.include_unverified_in_rumor)
    sources = parse_source_tweets(source_path)

    texts: List[str] = []
    ys: List[str] = []
    graph_feats: List[np.ndarray] = []
    time_feats: List[np.ndarray] = []

    matched = 0
    missing_tree = 0
    missing_source = 0
    parse_fail = 0

    for event_id, y in labels.items():
        tree_path = tree_dir / f"{event_id}.txt"
        text = sources.get(event_id)
        if text is None:
            missing_source += 1
            continue
        if not tree_path.exists():
            missing_tree += 1
            continue
        try:
            tree = parse_tree_file(tree_path)
        except Exception:
            parse_fail += 1
            continue
        texts.append(text)
        ys.append(y)
        graph_feats.append(tree_structural_features(tree))
        time_feats.append(hawkes_temporal_features(tree, cfg.hawkes_decay, cfg.hawkes_topk))
        matched += 1

    if matched == 0:
        raise RuntimeError(
            "No usable samples found. Check dataset_dir, label.txt/source_tweets.txt IDs, and tree filenames."
        )

    print("\n===== Dataset summary =====")
    print(f"Usable samples    : {matched}")
    print(f"Missing source txt: {missing_source}")
    print(f"Missing tree file : {missing_tree}")
    print(f"Parse failures    : {parse_fail}")
    print(f"Avg text length   : {np.mean([len(t.split()) for t in texts]):.2f} tokens")
    print(f"Label distribution: {dict(Counter(ys))}")

    return texts, ys, np.vstack(graph_feats), np.vstack(time_feats)


# -----------------------------
# Modeling utilities
# -----------------------------
def encode_labels(labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[x] for x in labels], dtype=np.int64)
    return y, classes


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    return {"acc": acc, "macro_f1": macro_f1}


def fit_eval_model(
    exp_name: str,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_dev: sparse.csr_matrix,
    y_dev: np.ndarray,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    classes: List[str],
    cfg: Config,
) -> Dict[str, float]:
    clf = LogisticRegression(
        C=cfg.clf_C,
        solver=cfg.clf_solver,
        max_iter=cfg.clf_max_iter,
        random_state=cfg.random_state,
    )
    clf.fit(X_train, y_train)
    dev_pred = clf.predict(X_dev)
    test_pred = clf.predict(X_test)
    dev_metrics = evaluate(f"{exp_name} / DEV", y_dev, dev_pred, classes)
    test_metrics = evaluate(f"{exp_name} / TEST", y_test, test_pred, classes)
    return {
        "model": exp_name,
        "nfeat": X_train.shape[1],
        "dev_acc": dev_metrics["acc"],
        "dev_f1": dev_metrics["macro_f1"],
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["macro_f1"],
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Twitter15/Twitter16 tree-time Hawkes rumor detection")
    ap.add_argument("--dataset_dir", type=str, required=True, help="Path to twitter15 or twitter16 directory")
    ap.add_argument("--task_mode", type=str, default="binary", choices=["binary", "four_class"])
    ap.add_argument("--include_unverified_in_rumor", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hawkes_decay", type=float, default=0.08)
    ap.add_argument("--hawkes_topk", type=int, default=256)
    ap.add_argument("--tfidf_max_features", type=int, default=6000)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--dev_size_from_train", type=float, default=0.15)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--output_json", type=Path, help="Save config and metrics to JSON")
    args = ap.parse_args()

    cfg = Config(
        task_mode=args.task_mode,
        include_unverified_in_rumor=args.include_unverified_in_rumor,
        hawkes_decay=args.hawkes_decay,
        hawkes_topk=args.hawkes_topk,
        tfidf_max_features=args.tfidf_max_features,
        test_size=args.test_size,
        dev_size_from_train=args.dev_size_from_train,
        random_state=args.random_state,
    )

    dataset_dir = Path(args.dataset_dir)
    print("Resolved config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print(f"\nUsing dataset_dir: {dataset_dir}")

    texts, labels, graph_feat, time_feat = build_dataset(dataset_dir, cfg)
    y, classes = encode_labels(labels)

    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    train_idx, dev_idx = train_test_split(
        train_idx,
        test_size=cfg.dev_size_from_train,
        random_state=cfg.random_state,
        stratify=y[train_idx],
    )

    texts_train = [texts[i] for i in train_idx]
    texts_dev = [texts[i] for i in dev_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train, y_dev, y_test = y[train_idx], y[dev_idx], y[test_idx]

    # text features
    print("\nFitting TF-IDF...")
    vec = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        sublinear_tf=cfg.tfidf_sublinear_tf,
        lowercase=True,
    )
    X_text_train = vec.fit_transform(texts_train)
    X_text_dev = vec.transform(texts_dev)
    X_text_test = vec.transform(texts_test)

    # scale dense features using train stats only
    scaler_graph = StandardScaler()
    scaler_time = StandardScaler()
    X_graph_train = scaler_graph.fit_transform(graph_feat[train_idx])
    X_graph_dev = scaler_graph.transform(graph_feat[dev_idx])
    X_graph_test = scaler_graph.transform(graph_feat[test_idx])
    X_time_train = scaler_time.fit_transform(time_feat[train_idx])
    X_time_dev = scaler_time.transform(time_feat[dev_idx])
    X_time_test = scaler_time.transform(time_feat[test_idx])

    X_graph_train_sp = sparse.csr_matrix(X_graph_train)
    X_graph_dev_sp = sparse.csr_matrix(X_graph_dev)
    X_graph_test_sp = sparse.csr_matrix(X_graph_test)
    X_time_train_sp = sparse.csr_matrix(X_time_train)
    X_time_dev_sp = sparse.csr_matrix(X_time_dev)
    X_time_test_sp = sparse.csr_matrix(X_time_test)

    experiments = {
        "text_only": (X_text_train, X_text_dev, X_text_test),
        "tree_only": (X_graph_train_sp, X_graph_dev_sp, X_graph_test_sp),
        "time_only": (X_time_train_sp, X_time_dev_sp, X_time_test_sp),
        "text_tree": (
            sparse.hstack([X_text_train, X_graph_train_sp]).tocsr(),
            sparse.hstack([X_text_dev, X_graph_dev_sp]).tocsr(),
            sparse.hstack([X_text_test, X_graph_test_sp]).tocsr(),
        ),
        "text_time": (
            sparse.hstack([X_text_train, X_time_train_sp]).tocsr(),
            sparse.hstack([X_text_dev, X_time_dev_sp]).tocsr(),
            sparse.hstack([X_text_test, X_time_test_sp]).tocsr(),
        ),
        "text_tree_time_hawkes": (
            sparse.hstack([X_text_train, X_graph_train_sp, X_time_train_sp]).tocsr(),
            sparse.hstack([X_text_dev, X_graph_dev_sp, X_time_dev_sp]).tocsr(),
            sparse.hstack([X_text_test, X_graph_test_sp, X_time_test_sp]).tocsr(),
        ),
        "no_time": (
            sparse.hstack([X_text_train, X_graph_train_sp]).tocsr(),
            sparse.hstack([X_text_dev, X_graph_dev_sp]).tocsr(),
            sparse.hstack([X_text_test, X_graph_test_sp]).tocsr(),
        ),
        "no_tree": (
            sparse.hstack([X_text_train, X_time_train_sp]).tocsr(),
            sparse.hstack([X_text_dev, X_time_dev_sp]).tocsr(),
            sparse.hstack([X_text_test, X_time_test_sp]).tocsr(),
        ),
    }

    print("\nPlanned experiments:")
    for name in experiments:
        print(f"  - {name}")

    summary = []
    for name, (xa, xb, xc) in experiments.items():
        print("\n" + "#" * 26 + f" Running: {name} " + "#" * 26)
        print(f"Feature dimension: {xa.shape[1]}")
        summary.append(
            fit_eval_model(name, xa, y_train, xb, y_dev, xc, y_test, classes, cfg)
        )

    summary = sorted(summary, key=lambda r: r["dev_acc"], reverse=True)
    print("\n" + "=" * 100)
    print("Experiment Summary (sorted by Validation Accuracy)")
    print("=" * 100)
    print(f"{'Model':<24}{'#Feat':>8}{'DevAcc':>12}{'DevF1':>11}{'TestAcc':>12}{'TestF1':>11}")
    print("-" * 100)
    for row in summary:
        print(
            f"{row['model']:<24}{row['nfeat']:>8d}{row['dev_acc']:>12.4f}{row['dev_f1']:>11.4f}"
            f"{row['test_acc']:>12.4f}{row['test_f1']:>11.4f}"
        )

    if args.output_json:
        import json
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"config": cfg.__dict__, "classes": classes, "experiments": summary}, indent=2), encoding="utf-8")

    best_baseline = next((r for r in summary if r["model"] == "text_tree"), None)
    full = next((r for r in summary if r["model"] == "text_tree_time_hawkes"), None)
    if best_baseline and full:
        print("\nComparison note:")
        print(
            f"  text_tree_time_hawkes improvement over text_tree: "
            f"{full['test_acc'] - best_baseline['test_acc']:+.4f}"
        )


if __name__ == "__main__":
    main()
