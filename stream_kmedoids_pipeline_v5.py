# stream_kmedoids_pipeline.py
"""
Streaming / Baseline K-medoids (mixed-type) with:
- Streaming over chunks with per-chunk weighted Gower DMs
- Summary-guided Hungarian matching across chunks
- Candidate pools per global cluster + Hard-point reservoir
- FINAL: "fullD" (N×N) or "coreset" refinement
- Memory-light final assignment O(N·K)
- Feature weights: manual or learned ("supervised" | "stability")  <-- updated
- Adaptive chunk sizing by RAM target
- One-shot fullD baseline
- Experiment logging to runs/<timestamp>/
- Time budget guard + preflight guard for final refinement
- Optional scikit-learn-extra final refinement (capped iterations)
- Progress-aware distance builders + optional memmap
- FullD ETA + memory guard prior to build
- NEW: learned weights saved to learned_weights.json + top weights in metrics
"""

import argparse, os, yaml, time, psutil, math, threading, json, pathlib, datetime, sys
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from pyclustering.cluster.kmedoids import kmedoids

try:
    from sklearn_extra.cluster import KMedoids as SKKMedoids
    HAVE_SKEXTRA = True
except Exception:
    HAVE_SKEXTRA = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------------- YAML config loader ----------------
def load_yaml_config(path: str | None) -> dict:
    defaults = {
        "csv_path": "synthetic_asthma_10k.csv",
        "k_fixed": 10,

        # core streaming
        "use_streaming": True,
        "chunk_size": 2000,
        "cand_per_local": 50,

        # final mode
        "final_mode": "coreset",         # "fullD" | "coreset"
        "max_coreset": 20000,

        # metrics
        "sil_sample": 3000,

        # weights
        "use_weights": False,
        "group_weights": {"numeric": 1.0, "binary": 1.0, "categorical": 1.0},
        "per_feature_weights": {},
        "weight_learning_mode": "none",  # "none" | "supervised" | "stability"
        "weight_learning_sample": 8000,
        "weight_learning_min_weight": 0.2,

        # matching & pools
        "use_candidate_pools": True,
        "use_hungarian": True,
        "summary_weight": 0.4,

        # reservoir
        "enable_hard_reservoir": False,
        "reservoir_mode": "quantile",
        "reservoir_quantile": 0.85,
        "reservoir_topk": 10,

        # adaptive chunking
        "auto_chunk": False,
        "ram_target_gb": 4.0,
        "ram_headroom_gb": 1.0,

        # time budget & guards
        "time_budget_secs": 0,
        "min_secs_before_final": 0,

        # optional final solver cap
        "use_sklearn_extra_final": False,
        "final_max_iter": 30,

        # progress & memmap flags
        "show_progress": True,
        "fullD_use_memmap": False,
        "core_use_memmap": False,

        "verbose": True,
        "heartbeat_secs": 15,
        "runs_dir": "runs",

        # schema
        "binary_cols": [
            "smoking","eosinophilia","neutrophilia","allergy","severity",
            "drug_obesity","drug_hypertension","drug_dyslipidemia","drug_diabetes"
        ],
        "num_cols": ["age","bmi"],
        "cat_cols": ["ethnicity"],
    }
    if path is None or not os.path.exists(path):
        if path:
            print(f"[Config] Not found: {path} — using defaults.")
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = defaults | {k: v for k, v in user_cfg.items() if k not in {"group_weights","per_feature_weights"}}
    cfg["group_weights"] = {**defaults["group_weights"], **user_cfg.get("group_weights", {})}
    cfg["per_feature_weights"] = {**defaults["per_feature_weights"], **user_cfg.get("per_feature_weights", {})}
    return cfg

_parser = argparse.ArgumentParser()
_parser.add_argument("--config", type=str, default="config.yml")
_args = _parser.parse_args()
CFG = load_yaml_config(_args.config)

# Bind config
CSV_PATH        = CFG["csv_path"]
K_FIXED         = int(CFG["k_fixed"])
CHUNK_SIZE      = int(CFG["chunk_size"])
CAND_PER_LOCAL  = int(CFG["cand_per_local"])
FINAL_MODE      = str(CFG["final_mode"]).lower()
MAX_CORESET     = int(CFG["max_coreset"])
SIL_SAMPLE      = None if CFG["sil_sample"] in (None, "null") else int(CFG["sil_sample"])
GROUP_WEIGHTS   = CFG["group_weights"]
PER_FEATURE_WEIGHTS = CFG["per_feature_weights"]
VERBOSE         = bool(CFG["verbose"])
HEARTBEAT_SECS  = int(CFG["heartbeat_secs"])
USE_STREAMING   = bool(CFG["use_streaming"])
USE_WEIGHTS     = bool(CFG["use_weights"])
USE_HUNGARIAN   = bool(CFG["use_hungarian"])
USE_CAND_POOLS  = bool(CFG["use_candidate_pools"])
SUMMARY_W       = float(CFG["summary_weight"])
RUNS_DIR        = CFG["runs_dir"]
TIME_BUDGET_SECS = int(CFG.get("time_budget_secs", 0))
MIN_SECS_BEFORE_FINAL = float(CFG.get("min_secs_before_final", 0))
USE_SKEXTRA_FINAL = bool(CFG.get("use_sklearn_extra_final", False))
FINAL_MAX_ITER  = int(CFG.get("final_max_iter", 30))
SHOW_PROGRESS   = bool(CFG.get("show_progress", True))
FULLD_MEMMAP    = bool(CFG.get("fullD_use_memmap", False))
CORE_MEMMAP     = bool(CFG.get("core_use_memmap", False))

START_TIME = time.perf_counter()
PEAK_RSS_GB = 0.0
DEADLINE = time.time() + TIME_BUDGET_SECS if TIME_BUDGET_SECS > 0 else None
def time_left_ok(): return (DEADLINE is None) or (time.time() < DEADLINE)
def secs_left():    return float("inf") if DEADLINE is None else (DEADLINE - time.time())

binary_cols = CFG["binary_cols"]; num_cols = CFG["num_cols"]; cat_cols = CFG["cat_cols"]
all_cols = binary_cols + num_cols + cat_cols

# Adaptive chunk sizing
def choose_chunk_size_from_ram(ram_target_gb: float, headroom_gb: float = 1.0, dtype_bytes: int = 4,
                               max_chunk: int | None = None, min_chunk: int = 500) -> int:
    target_bytes = max((ram_target_gb - headroom_gb), 0.25) * (1024**3)
    s_est = int(np.floor(np.sqrt(target_bytes / max(dtype_bytes, 1))))
    if max_chunk is not None: s_est = min(s_est, int(max_chunk))
    return max(s_est, min_chunk)

if CFG.get("auto_chunk", False):
    CHUNK_SIZE = choose_chunk_size_from_ram(CFG.get("ram_target_gb", 4.0), CFG.get("ram_headroom_gb", 1.0), 4, CHUNK_SIZE, 500)
    if VERBOSE: print(f"[AutoChunk] CHUNK_SIZE → {CHUNK_SIZE} rows")

# Pyclustering data types
try:
    from pyclustering.utils.data_type import type_data as _type_data
    DATA_TYPE_POINTS = _type_data.POINTS
    DATA_TYPE_DM     = _type_data.DISTANCE_MATRIX
except Exception:
    try:
        from pyclustering.utils import type_data as _type_data
        DATA_TYPE_POINTS = _type_data.POINTS
        DATA_TYPE_DM     = _type_data.DISTANCE_MATRIX
    except Exception:
        DATA_TYPE_POINTS = "points"; DATA_TYPE_DM = "distance_matrix"

# Profiling / Heartbeat
PROFILE = defaultdict(list); PROC = psutil.Process(os.getpid())
CURRENT_STAGE = "starting"; _stop_hb = threading.Event()
def mem_gb(): return PROC.memory_info().rss / (1024**3)

@contextmanager
def track(name: str):
    t0, m0 = time.perf_counter(), mem_gb()
    try: yield
    finally:
        PROFILE["step"].append(name)
        PROFILE["secs"].append(time.perf_counter() - t0)
        PROFILE["dmem_gb"].append(mem_gb() - m0)
        global PEAK_RSS_GB; PEAK_RSS_GB = max(PEAK_RSS_GB, mem_gb())

def _heartbeat():
    if not VERBOSE: return
    t0 = time.perf_counter()
    while not _stop_hb.wait(HEARTBEAT_SECS):
        rss = mem_gb(); elapsed = time.perf_counter() - t0
        global PEAK_RSS_GB; PEAK_RSS_GB = max(PEAK_RSS_GB, rss)
        print(f"[HB] {elapsed/60:5.1f} min | stage: {CURRENT_STAGE} | RSS ~{rss:.2f} GB | secs left ~{secs_left():.0f}", flush=True)

# ---------------- Encoding & weights ----------------
def encode_dataframe(df: pd.DataFrame):
    enc = df.copy()
    for c in cat_cols: enc[c] = enc[c].astype("category").cat.codes.astype(int)
    for c in binary_cols: enc[c] = enc[c].astype(int)
    for c in num_cols:    enc[c] = enc[c].astype(float)
    X = enc[all_cols].to_numpy()
    idx_bin = np.array([all_cols.index(c) for c in binary_cols], dtype=int)
    idx_num = np.array([all_cols.index(c) for c in num_cols], dtype=int)
    idx_cat = np.array([all_cols.index(c) for c in cat_cols], dtype=int)
    rng = np.maximum(enc[num_cols].max().to_numpy() - enc[num_cols].min().to_numpy(), 1e-9).astype(float)
    return X, idx_bin, idx_num, idx_cat, rng

def build_feature_weights_arrays(per_feature_weights: dict, group_weights: dict):
    """Return w_num, w_bin, w_cat arrays (aligned to num_cols, binary_cols, cat_cols) from a per-feature dict (+ group mult)."""
    gw = group_weights or {'numeric':1,'binary':1,'categorical':1}
    perw = {c: per_feature_weights.get(c, 1.0) for c in all_cols}
    for c in num_cols:  perw[c] *= gw.get('numeric', 1.0)
    for c in binary_cols: perw[c] *= gw.get('binary', 1.0)
    for c in cat_cols: perw[c] *= gw.get('categorical', 1.0)
    w_num = np.array([perw[c] for c in num_cols], float) if len(num_cols) else np.array([])
    w_bin = np.array([perw[c] for c in binary_cols], float) if len(binary_cols) else np.array([])
    w_cat = np.array([perw[c] for c in cat_cols], float) if len(cat_cols) else np.array([])
    eps = 1e-12
    return np.maximum(w_num, eps), np.maximum(w_bin, eps), np.maximum(w_cat, eps), perw

# -------- Weighted Gower (vector ops) --------
def pairwise_gower_block_w(X, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat,
                           num_block=512, bin_block=2048, cat_block=2048, **_ignore):
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32); total_w = 0.0
    if idx_num.size:
        Xn = X[:, idx_num].astype(np.float64, copy=False) / rng
        wn = w_num.astype(np.float64, copy=False); total_w += float(wn.sum())
        for a in range(0, n, num_block):
            A = Xn[a:a+num_block]
            diff = np.abs(A[:, None, :] - Xn[None, :, :]) * wn
            D[a:a+num_block] += diff.sum(axis=2)
    if idx_bin.size:
        Xb = X[:, idx_bin].astype(np.uint8, copy=False)
        wb = w_bin.astype(np.float64, copy=False); total_w += float(wb.sum())
        for a in range(0, n, bin_block):
            A = Xb[a:a+bin_block]
            diff = (A[:, None, :] != Xb[None, :, :]).astype(np.float64) * wb
            D[a:a+bin_block] += diff.sum(axis=2)
    if idx_cat.size:
        Xc = X[:, idx_cat].astype(np.int32, copy=False)
        wc = w_cat.astype(np.float64, copy=False); total_w += float(wc.sum())
        for a in range(0, n, cat_block):
            A = Xc[a:a+cat_block]
            diff = (A[:, None, :] != Xc[None, :, :]).astype(np.float64) * wc
            D[a:a+cat_block] += diff.sum(axis=2)
    if total_w <= 0: total_w = 1.0
    D /= total_w
    D = np.maximum(D, D.T); np.fill_diagonal(D, 0.0)
    return D

# -------- Progress-aware + memmap builder --------
def pairwise_gower_block_w_with_progress(
    X, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat,
    num_block=512, bin_block=2048, cat_block=2048,
    desc="Build D", use_memmap=False, memmap_path=None, dtype=np.float32
):
    n = X.shape[0]
    if use_memmap:
        if memmap_path is None:
            raise ValueError("memmap_path required when use_memmap=True")
        D = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(n, n))
    else:
        D = np.zeros((n, n), dtype=dtype)

    total_w = 0.0

    if idx_num.size:
        Xn = X[:, idx_num].astype(np.float64, copy=False) / rng
        wn = w_num.astype(np.float64, copy=False); total_w += float(wn.sum())
        it = range(0, n, num_block)
        if tqdm is not None and SHOW_PROGRESS: it = tqdm(it, desc=f"{desc} [num]", unit="blk")
        for a in it:
            A = Xn[a:a+num_block]
            diff = np.abs(A[:, None, :] - Xn[None, :, :]) * wn
            D[a:a+num_block] += diff.sum(axis=2)
            if use_memmap: D.flush()

    if idx_bin.size:
        Xb = X[:, idx_bin].astype(np.uint8, copy=False)
        wb = w_bin.astype(np.float64, copy=False); total_w += float(wb.sum())
        it = range(0, n, bin_block)
        if tqdm is not None and SHOW_PROGRESS: it = tqdm(it, desc=f"{desc} [bin]", unit="blk")
        for a in it:
            A = Xb[a:a+bin_block]
            diff = (A[:, None, :] != Xb[None, :, :]).astype(np.float64) * wb
            D[a:a+bin_block] += diff.sum(axis=2)
            if use_memmap: D.flush()

    if idx_cat.size:
        Xc = X[:, idx_cat].astype(np.int32, copy=False)
        wc = w_cat.astype(np.float64, copy=False); total_w += float(wc.sum())
        it = range(0, n, cat_block)
        if tqdm is not None and SHOW_PROGRESS: it = tqdm(it, desc=f"{desc} [cat]", unit="blk")
        for a in it:
            A = Xc[a:a+cat_block]
            diff = (A[:, None, :] != Xc[None, :, :]).astype(np.float64) * wc
            D[a:a+cat_block] += diff.sum(axis=2)
            if use_memmap: D.flush()

    if total_w <= 0: total_w = 1.0
    D[:] = D / total_w

    # Symmetrize & zero diagonal in-place by blocks (to save memory)
    bs = 4096
    it = range(0, n, bs)
    if tqdm is not None and SHOW_PROGRESS: it = tqdm(it, desc=f"{desc} [sym/diag]", unit="blk")
    for a in it:
        b = min(a+bs, n)
        D[a:b, :] = np.maximum(D[a:b, :], D[:, a:b].T)
        np.fill_diagonal(D[a:b, a:b], 0.0)
        if use_memmap: D.flush()

    return D

def estimate_fullD_runtime(X, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat, probe_rows=1024):
    t0 = time.perf_counter()
    Xp = X[:probe_rows]
    _ = pairwise_gower_block_w(Xp, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat)
    dt = time.perf_counter() - t0
    n = X.shape[0]
    est = dt * (n / probe_rows) ** 2
    return est

# Seeding (vectorised K-medoids++)
def gower_to_vector_w(X, v, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat):
    total_w, out = 0.0, None
    if idx_num.size:
        dn = (np.abs(X[:, idx_num] - v[idx_num]) / rng) * w_num
        dn = dn.sum(axis=1); out = dn if out is None else out + dn; total_w += float(w_num.sum())
    if idx_bin.size:
        db = (X[:, idx_bin] != v[idx_bin]).astype(np.float64) * w_bin
        db = db.sum(axis=1); out = db if out is None else out + db; total_w += float(w_bin.sum())
    if idx_cat.size:
        dc = (X[:, idx_cat] != v[idx_cat]).astype(np.float64) * w_cat
        dc = dc.sum(axis=1); out = (out + dc) if out is not None else dc; total_w += float(w_cat.sum())
    return (out / max(total_w, 1.0)).astype(np.float64) if out is not None else np.zeros(X.shape[0], float)

def kmedoidspp_init_points_vec_w(X, K, idx_num, idx_bin, idx_cat, rng_vec, w_num, w_bin, w_cat, seed=0):
    rng = np.random.default_rng(seed); n = X.shape[0]
    medoids = [int(rng.integers(0, n))]
    dmin = gower_to_vector_w(X, X[medoids[0]], idx_num, idx_bin, idx_cat, rng_vec, w_num, w_bin, w_cat)
    for _ in range(1, K):
        p = dmin**2; s = p.sum()
        cand = int(np.argmax(dmin)) if s <= 0 else int(rng.choice(n, p=p/s))
        medoids.append(cand)
        d_cand = gower_to_vector_w(X, X[cand], idx_num, idx_bin, idx_cat, rng_vec, w_num, w_bin, w_cat)
        dmin = np.minimum(dmin, d_cand)
    return medoids

# Summaries & matching
def cluster_summary(X_df: pd.DataFrame) -> np.ndarray:
    cont = X_df[num_cols].mean().to_numpy(float) if len(num_cols) else np.array([], float)
    binv = X_df[binary_cols].mean().to_numpy(float) if len(binary_cols) else np.array([], float)
    cat_codes = (X_df[cat_cols].astype("category").apply(lambda s: s.cat.codes).mode().iloc[0].to_numpy(int)
                 if len(cat_cols) else np.array([], int))
    return np.concatenate([cont, binv, cat_codes])

def summary_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    m, nb, nc = len(num_cols), len(binary_cols), len(cat_cols)
    d_cont = np.linalg.norm(s1[:m]-s2[:m]) if m else 0.0
    d_bin = float(np.mean(np.abs(s1[m:m+nb]-s2[m:m+nb]))) if nb else 0.0
    d_cat = float(np.mean(s1[m+nb:] != s2[m+nb:])) if nc else 0.0
    return d_cont + d_bin + d_cat

# Coreset, reservoir & assignment
def build_coreset_indices(global_cand_idx: dict, max_coreset: int | None = None) -> np.ndarray:
    pools = [np.array(v, dtype=int) for v in global_cand_idx.values() if len(v) > 0]
    if not pools: return np.array([], dtype=int)
    full = np.unique(np.concatenate(pools))
    if max_coreset is None or len(full) <= max_coreset: return full
    sizes = np.array([len(p) for p in pools], float); probs = sizes / sizes.sum()
    targets = np.maximum((probs * max_coreset).astype(int), 1)
    diff = max_coreset - targets.sum()
    if diff != 0:
        order = np.argsort(-probs); i = 0
        while diff != 0:
            idx = order[i % len(order)]
            if diff > 0: targets[idx] += 1; diff -= 1
            else:
                if targets[idx] > 1: targets[idx] -= 1; diff += 1
            i += 1
    rng = np.random.default_rng(0)
    sampled = [rng.choice(p, size=targets[i], replace=False) for i, p in enumerate(pools)]
    return np.unique(np.concatenate(sampled))

def update_hard_reservoir(global_hard_set: set, global_indices_members: np.ndarray,
                          medoid_to_members_dists: np.ndarray, mode="quantile", q=0.85, topk=10):
    if global_indices_members.size == 0: return
    if mode == "topk":
        k = max(1, min(topk, global_indices_members.size))
        sel = np.argpartition(medoid_to_members_dists, -k)[-k:]
    else:
        cutoff = np.quantile(medoid_to_members_dists, q)
        sel = np.where(medoid_to_members_dists >= cutoff)[0]
    for idx in (global_indices_members[sel].tolist()):
        global_hard_set.add(int(idx))

def assign_by_nearest_medoid_blocked(X_all, medoid_rows, idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat):
    N = X_all.shape[0]; best = np.full(N, np.inf, float); labels = np.zeros(N, int)
    for j in range(medoid_rows.shape[0]):
        dcol = gower_to_vector_w(X_all, medoid_rows[j], idx_num, idx_bin, idx_cat, rng, w_num, w_bin, w_cat)
        better = dcol < best; labels[better] = j; best[better] = dcol[better]
    return labels

# ---------------- Weight learning (updated, no caps; mean-normalized to 1.0) ----------------
def _cramers_v_from_table(tbl: np.ndarray) -> float:
    if tbl.size == 0: return 0.0
    r, c = tbl.shape
    if r < 2 or c < 2: return 0.0
    n = tbl.sum()
    if n <= 0: return 0.0
    row_sums = tbl.sum(axis=1, keepdims=True)
    col_sums = tbl.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((tbl - expected) ** 2 / np.where(expected == 0, 1, expected))
    denom = n * max(min(r - 1, c - 1), 1)
    return float(np.sqrt(max(chi2, 0.0) / denom))

def _eta_squared_anova(values: np.ndarray, labels: np.ndarray) -> float:
    mask = ~np.isnan(values)
    x = values[mask]; y = labels[mask]
    if x.size < 3 or len(np.unique(y)) < 2: return 0.0
    overall = x.mean()
    ssb = 0.0; ssw = 0.0
    for g in np.unique(y):
        xi = x[y == g]
        if xi.size == 0: continue
        mu = xi.mean()
        ssb += xi.size * (mu - overall) ** 2
        ssw += np.sum((xi - mu) ** 2)
    sst = ssb + ssw
    if sst <= 0: return 0.0
    return float(ssb / sst)

def _binary_effect(values: np.ndarray, labels: np.ndarray) -> float:
    v = values.astype(int)
    classes = np.unique(labels)
    tbl = np.zeros((2, len(classes)), dtype=float)
    for j, g in enumerate(classes):
        vg = v[labels == g]
        if vg.size == 0: continue
        tbl[0, j] = np.sum(vg == 0)
        tbl[1, j] = np.sum(vg == 1)
    return _cramers_v_from_table(tbl)

def learn_weights_supervised_per_feature(df: pd.DataFrame, y_true: np.ndarray,
                                         num_cols, binary_cols, cat_cols, min_w: float = 0.2) -> dict:
    feats, vals = [], []
    for c in num_cols:
        feats.append(c); vals.append(_eta_squared_anova(df[c].to_numpy(dtype=float), y_true))
    for c in binary_cols:
        feats.append(c); vals.append(_binary_effect(df[c].to_numpy(dtype=int), y_true))
    for c in cat_cols:
        feats.append(c)
        codes = df[c].astype("category").cat.codes.to_numpy()
        cats = np.unique(codes); classes = np.unique(y_true)
        tbl = np.zeros((len(cats), len(classes)), dtype=float)
        for i, cat in enumerate(cats):
            for j, cls in enumerate(classes):
                tbl[i, j] = np.sum((codes == cat) & (y_true == cls))
        vals.append(_cramers_v_from_table(tbl))
    vals = np.array(vals, dtype=float)
    vals = np.maximum(vals, min_w)                 # floor only (NO CAP)
    mean_v = vals.mean() if vals.size else 1.0     # normalize to mean 1.0
    vals = vals / (mean_v if mean_v > 0 else 1.0)
    return {f: float(w) for f, w in zip(feats, vals)}

def _kmedoids_labels_points_smallDM(D: np.ndarray, k: int, init: list[int] | None = None):
    if init is None:
        # trivial init: farthest-first on DM
        n = D.shape[0]; rng = np.random.default_rng(0)
        seeds = [int(rng.integers(0, n))]
        dmin = D[seeds[0]].copy()
        while len(seeds) < k:
            cand = int(np.argmax(dmin))
            seeds.append(cand)
            dmin = np.minimum(dmin, D[cand])
        init = seeds
    algo = kmedoids(D, init, data_type="distance_matrix", ccore=True)
    algo.process()
    clusters = algo.get_clusters()
    labels = np.empty(D.shape[0], dtype=int)
    for cid, idxs in enumerate(clusters): labels[idxs] = cid
    return labels

def learn_weights_stability_per_feature(
    df: pd.DataFrame, k_fixed: int,
    num_cols, binary_cols, cat_cols,
    sample_n: int = 4000,      # smaller default: much faster, still stable
    seed: int = 0,
    min_w: float = 0.2
) -> dict:
    """
    Fast stability weighting:
      - Build per-feature distance contributions once: d_j
      - S = sum_j d_j
      - For each feature j, D_minus = (S - d_j) / (p - 1)
      - Fix base medoids from PAM(D_uniform), then nearest-medoid assignment on D_minus
      - Weight = max(min_w, 1 - NMI(labels_base, labels_minus_j)), mean-normalized
    """
    rng = np.random.default_rng(seed)
    cols_all = list(num_cols) + list(binary_cols) + list(cat_cols)
    p = len(cols_all)

    sub = df.sample(n=min(sample_n, len(df)), random_state=seed).reset_index(drop=True)

    # ---- Encode once
    enc = sub.copy()
    for c in cat_cols: enc[c] = enc[c].astype("category").cat.codes.astype(int)
    for c in binary_cols: enc[c] = enc[c].astype(int)
    for c in num_cols:    enc[c] = enc[c].astype(float)
    X = enc[cols_all].to_numpy()

    # index arrays
    idx_num = np.array([cols_all.index(c) for c in num_cols], dtype=int)
    idx_bin = np.array([cols_all.index(c) for c in binary_cols], dtype=int)
    idx_cat = np.array([cols_all.index(c) for c in cat_cols], dtype=int)

    # numeric ranges
    if len(num_cols):
        num_rng = np.maximum(
            enc[num_cols].max().to_numpy() - enc[num_cols].min().to_numpy(),
            1e-9
        ).astype(float)
    else:
        num_rng = np.array([], dtype=float)

    n = X.shape[0]
    d_sum = np.zeros((n, n), dtype=np.float32)
    d_parts: list[tuple[str, np.ndarray]] = []

    # ---- Per-feature contributions: d_j
    # Numeric: |x_i - x_j| / range
    for i, c in enumerate(num_cols):
        col = X[:, idx_num[i]].astype(np.float64, copy=False) / num_rng[i]
        diff = np.abs(col[:, None] - col[None, :]).astype(np.float32, copy=False)
        np.fill_diagonal(diff, 0.0)
        d_parts.append((c, diff)); d_sum += diff

    # Binary: 1 if different, else 0
    for i, c in enumerate(binary_cols):
        col = X[:, idx_bin[i]].astype(np.int32, copy=False)
        diff = (col[:, None] != col[None, :]).astype(np.float32, copy=False)
        np.fill_diagonal(diff, 0.0)
        d_parts.append((c, diff)); d_sum += diff

    # Categorical: 1 if different, else 0
    for i, c in enumerate(cat_cols):
        col = X[:, idx_cat[i]].astype(np.int32, copy=False)
        diff = (col[:, None] != col[None, :]).astype(np.float32, copy=False)
        np.fill_diagonal(diff, 0.0)
        d_parts.append((c, diff)); d_sum += diff

    if p <= 1:
        return {name: 1.0 for name, _ in d_parts}

    D_uniform = d_sum / float(p)
    # ensure symmetry + zero diag (should already hold, but cheap & safe)
    D_uniform = np.maximum(D_uniform, D_uniform.T); np.fill_diagonal(D_uniform, 0.0)

    # ---- PAM with explicit farthest-first seeding (NO None!)
    def _pam_labels_and_medoids(D: np.ndarray, k: int, seed: int = 0):
        n = D.shape[0]
        r = np.random.default_rng(seed)
        seeds = [int(r.integers(0, n))]
        dmin = D[seeds[0]].copy()
        while len(seeds) < k:
            cand = int(np.argmax(dmin))
            seeds.append(cand)
            dmin = np.minimum(dmin, D[cand])
        algo = kmedoids(D, seeds, data_type="distance_matrix", ccore=True)
        algo.process()
        clusters = algo.get_clusters()
        medoids = algo.get_medoids()
        labels = np.empty(n, dtype=int)
        for cid, idxs in enumerate(clusters):
            labels[idxs] = cid
        return labels, np.array(medoids, dtype=int)

    labels_base, medoids = _pam_labels_and_medoids(D_uniform, k_fixed, seed=seed)

    # ---- For each feature, form D^{(-j)} and reassign to fixed medoids
    from sklearn.metrics import normalized_mutual_info_score as NMI
    drops = np.zeros(p, dtype=np.float64)

    for j, (name, d_j) in enumerate(d_parts):
        # (S - d_j) / (p - 1)
        D_minus = (d_sum - d_j) / float(p - 1)
        # symmetry & diag safety
        D_minus = np.maximum(D_minus, D_minus.T); np.fill_diagonal(D_minus, 0.0)
        # nearest medoid using distances to medoid columns
        Dm = D_minus[:, medoids]                 # (n, k)
        labels2 = np.argmin(Dm, axis=1)
        drops[j] = max(0.0, 1.0 - NMI(labels_base, labels2))

    # Floor + mean-normalize
    drops = np.maximum(drops, float(min_w))
    mean_d = float(drops.mean()) if drops.size else 1.0
    weights = drops / (mean_d if mean_d > 0 else 1.0)

    return {name: float(w) for (name, _), w in zip(d_parts, weights)}


# ---------------- Main ----------------
if __name__ == "__main__":
    hb_thread = threading.Thread(target=_heartbeat, daemon=True); hb_thread.start()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(RUNS_DIR) / run_id; outdir.mkdir(parents=True, exist_ok=True)

    def partial_save_and_exit(note, labels, medoid_rows, core_idx=None, y_true=None,
                              X_all=None, IDX_NUM=None, IDX_BIN=None, IDX_CAT=None, NUM_RNG=None,
                              w_num=None, w_bin=None, w_cat=None, used_per_feature=None):
        print(f"[TimeBudget] {note} Saving artifacts…", flush=True)
        if labels is None and (medoid_rows is not None) and (X_all is not None):
            labels = assign_by_nearest_medoid_blocked(X_all, medoid_rows, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)

        ari = nmi = sil = float("nan")
        if labels is not None and y_true is not None:
            try:
                ari = float(adjusted_rand_score(y_true, labels))
                nmi = float(normalized_mutual_info_score(y_true, labels))
            except Exception: pass
        if labels is not None and X_all is not None:
            try:
                ss = min(int(CFG.get("sil_sample") or 1500), len(X_all))
                idx = np.random.default_rng(1).choice(len(X_all), size=ss, replace=False)
                X_s = X_all[idx]
                D_s = pairwise_gower_block_w(X_s, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                sil = float(silhouette_score(D_s, labels[idx], metric="precomputed"))
            except Exception: pass

        with open(outdir / "config.json", "w") as f: json.dump(CFG, f, indent=2)
        total_secs = time.perf_counter() - START_TIME
        metrics = {"ARI": ari, "NMI": nmi, "Silhouette": sil,
                   "K": int(K_FIXED), "runtime_seconds": total_secs,
                   "peak_rss_gb": PEAK_RSS_GB, "note": note,
                   "final_mode": FINAL_MODE, "streaming": USE_STREAMING}
        if used_per_feature is not None:
            top = sorted(used_per_feature.items(), key=lambda kv: kv[1], reverse=True)[:8]
            metrics["top_feature_weights"] = [{"feat": k, "w": float(v)} for k, v in top]
        with open(outdir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
        if labels is not None:
            pd.Series(labels, name="label").to_csv(outdir / "labels.csv", index=False)
        if core_idx is not None:
            np.save(outdir / "coreset_idx_partial.npy", core_idx)
        pd.DataFrame(PROFILE).to_csv(outdir / "profile.csv", index=False)
        if used_per_feature is not None:
            with open(outdir / "learned_weights.json", "w") as f:
                json.dump({"per_feature_weights": used_per_feature}, f, indent=2)
        print(f"[Saved] partial artifacts → {outdir}", flush=True)
        _stop_hb.set()
        try: threading.Thread.join(hb_thread, timeout=1)
        except Exception: pass
        sys.exit(0)

    try:
        # Load & tidy
        with track("load_csv"):
            df = pd.read_csv(CSV_PATH)
        with track("cast_dtypes"):
            for c in binary_cols: df[c] = df[c].astype(int)
            for c in num_cols:    df[c] = df[c].astype(float)
            if "ethnicity" in df.columns: df["ethnicity"] = df["ethnicity"].astype("category")
            y_true = df["true_cluster"].to_numpy() if "true_cluster" in df.columns else None

        # Encode once
        with track("encode_dataframe"):
            X_all, IDX_BIN, IDX_NUM, IDX_CAT, NUM_RNG = encode_dataframe(df[all_cols])

        # ---------------- Learn / build weights (updated) ----------------
        perw_learned = None
        mode_w = CFG.get("weight_learning_mode", "none")
        if CFG.get("use_weights", False) and mode_w == "supervised" and ("true_cluster" in df.columns):
            if not time_left_ok():
                partial_save_and_exit("Budget hit before supervised weight learning.", None, None,
                                      y_true=y_true, X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG)
            with track("learn_weights_supervised"):
                m = min(int(CFG.get("weight_learning_sample", 8000)), len(df))
                sub_idx = np.random.default_rng(0).choice(len(df), size=m, replace=False)
                perw_learned = learn_weights_supervised_per_feature(
                    df.iloc[sub_idx], df["true_cluster"].to_numpy()[sub_idx],
                    num_cols, binary_cols, cat_cols,
                    CFG.get("weight_learning_min_weight", 0.2)
                )
        elif CFG.get("use_weights", False) and mode_w == "stability":
            if not time_left_ok():
                partial_save_and_exit("Budget hit before stability weight learning.", None, None,
                                      y_true=y_true, X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG)
            with track("learn_weights_stability"):
                perw_learned = learn_weights_stability_per_feature(
                    df, K_FIXED, num_cols, binary_cols, cat_cols,
                    sample_n=CFG.get("weight_learning_sample", 8000),
                    seed=42, min_w=CFG.get("weight_learning_min_weight", 0.2)
                )

        # Merge priority: learned → manual per-feature → group
        perw_final = {c: 1.0 for c in all_cols}
        if perw_learned is not None: perw_final.update(perw_learned)
        perw_final.update(PER_FEATURE_WEIGHTS or {})
        # convert to arrays aligned to column groups
        w_num, w_bin, w_cat, perw_used = build_feature_weights_arrays(perw_final, GROUP_WEIGHTS)

        # Persist learned/used weights
        try:
            with open(outdir / "learned_weights.json", "w") as f:
                json.dump({"per_feature_weights": perw_used, "learned_only": perw_learned}, f, indent=2)
        except Exception:
            pass

        if not USE_STREAMING:
            print("[Mode] One-shot baseline (no streaming)", flush=True)
            if not time_left_ok():
                partial_save_and_exit("Budget hit before baseline DM.", None, None,
                                      y_true=y_true, X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                      w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)
            with track("build_D_full_baseline"):
                D_full = (pairwise_gower_block_w_with_progress if SHOW_PROGRESS or FULLD_MEMMAP else pairwise_gower_block_w)(
                    X_all, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat,
                    desc="Build D_full (baseline)",
                    use_memmap=FULLD_MEMMAP,
                    memmap_path=str((outdir / "D_full.dat").resolve()) if FULLD_MEMMAP else None
                )
            with track("seed_baseline"):
                init = kmedoidspp_init_points_vec_w(X_all, K_FIXED, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat, seed=0)
            with track("kmedoids_full_baseline"):
                algo = kmedoids(D_full, init, data_type="distance_matrix", ccore=True); algo.process()
                clusters_final = algo.get_clusters()
            labels = np.empty(len(df), int)
            for cid, idxs in enumerate(clusters_final): labels[idxs] = cid

        else:
            # Streaming path
            TOTAL_CHUNKS = math.ceil(len(df) / CHUNK_SIZE)
            print(f"Data: {df.shape} | chunks: {TOTAL_CHUNKS} (size {CHUNK_SIZE}) | Peak RSS ~{mem_gb():.2f} GB", flush=True)
            global_medoid_indices = None
            global_cand_idx = {k: [] for k in range(K_FIXED)}
            global_summaries = {k: [] for k in range(K_FIXED)}
            global_hard_set = set()

            with track("chunk_loop"):
                it = range(0, len(df), CHUNK_SIZE)
                if tqdm is not None and VERBOSE: it = tqdm(it, desc="Chunks", unit="chunk")
                t0_all = time.perf_counter()

                for t, start in enumerate(it):
                    if not time_left_ok() and global_medoid_indices is not None:
                        medoid_rows = X_all[np.array(global_medoid_indices[:K_FIXED], int)]
                        partial_save_and_exit("Budget hit mid-chunk loop — assigned via current global medoids.",
                                              None, medoid_rows, y_true=y_true,
                                              X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                              w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

                    chunk_t0 = time.perf_counter()
                    stop = min(start + CHUNK_SIZE, len(df))
                    chunk = df.iloc[start:stop].reset_index(drop=True)
                    X_chunk = X_all[start:stop]

                    with track(f"chunk_{t}_buildD"):
                        D_chunk = (pairwise_gower_block_w_with_progress if SHOW_PROGRESS else pairwise_gower_block_w)(
                            X_chunk, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat,
                            desc=f"Build D_chunk#{t}"
                        )
                    with track(f"chunk_{t}_init"):
                        init_meds = kmedoidspp_init_points_vec_w(X_chunk, K_FIXED, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat, seed=0)
                    with track(f"chunk_{t}_kmedoids"):
                        algo = kmedoids(D_chunk, init_meds, data_type="distance_matrix", ccore=True)
                        algo.process()
                        clusters = algo.get_clusters(); medoid_ids = algo.get_medoids()

                    local_medoid_global = [start + m for m in medoid_ids]

                    if t == 0:
                        global_medoid_indices = local_medoid_global
                        if USE_CAND_POOLS:
                            with track(f"chunk_{t}_candidates"):
                                for cid, mid in enumerate(medoid_ids):
                                    members = clusters[cid]; dists = D_chunk[mid, members]
                                    order = np.argsort(dists)[:min(CAND_PER_LOCAL, len(members))]
                                    chosen = (start + np.array(members)[order]).tolist()
                                    global_cand_idx[cid].extend(chosen)
                                    global_summaries[cid].append(cluster_summary(chunk.iloc[members]))
                                    if CFG.get("enable_hard_reservoir", False):
                                        global_indices_members = start + np.array(members)
                                        update_hard_reservoir(global_hard_set, global_indices_members, dists,
                                                              CFG.get("reservoir_mode","quantile"),
                                                              CFG.get("reservoir_quantile",0.85),
                                                              CFG.get("reservoir_topk",10))
                        if VERBOSE:
                            elapsed = time.perf_counter() - chunk_t0
                            avg = (time.perf_counter() - t0_all) / (t + 1)
                            eta = avg * (TOTAL_CHUNKS - (t + 1))
                            print(f"[Chunk {t+1}/{TOTAL_CHUNKS}] {stop-start} rows | {elapsed:5.1f}s | avg {avg:5.1f}s | ETA {eta/60:5.1f}m | RSS {mem_gb():.2f} GB", flush=True)
                        continue

                    # Match local→global
                    if USE_HUNGARIAN:
                        with track(f"chunk_{t}_match_hungarian"):
                            K = K_FIXED
                            cost = np.zeros((K, K), float)
                            local_summ = [cluster_summary(chunk.iloc[idxs]) for idxs in clusters]
                            global_summ = [np.mean(global_summaries[g], axis=0) if len(global_summaries[g])>0 else local_summ[0]
                                           for g in range(K)]
                            alpha = 1.0 - SUMMARY_W
                            # distance from local medoid -> current global medoid
                            for i in range(K):
                                for j in range(K):
                                    d_med = gower_to_vector_w(X_all, X_all[global_medoid_indices[j]], IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)[local_medoid_global[i]]
                                    d_sum = summary_distance(local_summ[i], global_summ[j])
                                    cost[i, j] = alpha * d_med + (1.0 - alpha) * d_sum
                            row_ind, col_ind = linear_sum_assignment(cost)
                            mapping = dict(zip(row_ind, col_ind))
                    else:
                        with track(f"chunk_{t}_match_greedy"):
                            mapping, used = {}, set()
                            for i in range(K_FIXED):
                                dists = []
                                for j in range(K_FIXED):
                                    if j in used: dists.append((np.inf, j)); continue
                                    d_med = gower_to_vector_w(X_all, X_all[global_medoid_indices[j]], IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)[local_medoid_global[i]]
                                    dists.append((d_med, j))
                                jbest = min(dists, key=lambda x: x[0])[1]
                                mapping[i] = jbest; used.add(jbest)

                    # Append candidates
                    if USE_CAND_POOLS:
                        with track(f"chunk_{t}_candidates"):
                            for local_cid, global_cid in mapping.items():
                                mid = medoid_ids[local_cid]; members = clusters[local_cid]; dists = D_chunk[mid, members]
                                order = np.argsort(dists)[:min(CAND_PER_LOCAL, len(members))]
                                chosen = (start + np.array(members)[order]).tolist()
                                seen = set(global_cand_idx[global_cid])
                                for idx in chosen:
                                    if idx not in seen: global_cand_idx[global_cid].append(idx); seen.add(idx)
                                global_summaries[global_cid].append(cluster_summary(chunk.iloc[members]))
                                if CFG.get("enable_hard_reservoir", False):
                                    global_indices_members = start + np.array(members)
                                    update_hard_reservoir(global_hard_set, global_indices_members, dists,
                                                          CFG.get("reservoir_mode","quantile"),
                                                          CFG.get("reservoir_quantile",0.85),
                                                          CFG.get("reservoir_topk",10))

                    if VERBOSE:
                        elapsed = time.perf_counter() - chunk_t0
                        avg = (time.perf_counter() - t0_all) / (t + 1)
                        eta = avg * (TOTAL_CHUNKS - (t + 1))
                        print(f"[Chunk {t+1}/{TOTAL_CHUNKS}] {stop-start} rows | {elapsed:5.1f}s | avg {avg:5.1f}s | ETA {eta/60:5.1f}m | RSS {mem_gb():.2f} GB", flush=True)

            # Seeds from candidate pools (or fallback)
            with track("final_init_from_candidates"):
                init_meds_refined = []
                if USE_CAND_POOLS:
                    for k in range(K_FIXED):
                        cand = global_cand_idx[k]
                        if len(cand) == 0: continue
                        X_cand = X_all[cand]
                        avg = []
                        for i in range(len(cand)):
                            dv = gower_to_vector_w(X_cand, X_cand[i], IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                            avg.append(dv.sum() / max(len(cand) - 1, 1))
                        best_local = int(np.argmin(avg))
                        init_meds_refined.append(cand[best_local])
                if len(init_meds_refined) < K_FIXED:
                    n = X_all.shape[0]; rng = np.random.default_rng(123)
                    seeds = [int(rng.integers(0, n))]
                    dmin = gower_to_vector_w(X_all, X_all[seeds[0]], IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                    while len(seeds) < K_FIXED:
                        cand = int(np.argmax(dmin)); seeds.append(cand)
                        d_cand = gower_to_vector_w(X_all, X_all[cand], IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                        dmin = np.minimum(dmin, d_cand)
                    init_meds_refined = seeds[:K_FIXED]

            # Preflight time guard
            if (MIN_SECS_BEFORE_FINAL > 0 and secs_left() < MIN_SECS_BEFORE_FINAL) or not time_left_ok():
                medoid_rows = X_all[np.array(init_meds_refined, int)]
                partial_save_and_exit("Not enough time left for final refinement — seed-based assignment.",
                                      None, medoid_rows, y_true=y_true,
                                      X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                      w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

            # Final refinement
            if FINAL_MODE == "fulld":
                # ETA & Memory guard
                n = X_all.shape[0]
                bytes_f32 = n * n * 4
                bytes_f64_copy = n * n * 8
                avail = psutil.virtual_memory().available
                reserve = int(16 * (1024**3))  # keep 16 GiB free
                if VERBOSE:
                    est = estimate_fullD_runtime(X_all, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat, probe_rows=1024)
                    print(f"[FullD] ETA ~{est/60:.1f} min; D float32 ≈ {bytes_f32/2**30:.1f} GiB; "
                          f"worst-case with float64 copy ≈ {(bytes_f32+bytes_f64_copy)/2**30:.1f} GiB; "
                          f"avail ≈ {avail/2**30:.1f} GiB", flush=True)
                if avail - reserve < bytes_f32:
                    medoid_rows = X_all[np.array(init_meds_refined, int)]
                    partial_save_and_exit("Insufficient RAM for fullD build — seed-based assignment.",
                                          None, medoid_rows, y_true=y_true,
                                          X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                          w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)
                if avail - reserve < (bytes_f32 + bytes_f64_copy) and CFG.get("use_sklearn_extra_final", False):
                    print("[MemoryGuard] Disabling scikit-learn-extra final to avoid implicit float64 copy.", flush=True)
                    USE_SKEXTRA_FINAL = False

                with track("build_D_full"):
                    D_full = (pairwise_gower_block_w_with_progress if SHOW_PROGRESS or FULLD_MEMMAP else pairwise_gower_block_w)(
                        X_all, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat,
                        desc="Build D_full",
                        use_memmap=FULLD_MEMMAP,
                        memmap_path=str((outdir / "D_full.dat").resolve()) if FULLD_MEMMAP else None
                    )

                if not time_left_ok() or (MIN_SECS_BEFORE_FINAL > 0 and secs_left() < MIN_SECS_BEFORE_FINAL):
                    medoid_rows = X_all[np.array(init_meds_refined, int)]
                    partial_save_and_exit("Not enough time left for fullD refinement — seed-based assignment.",
                                          None, medoid_rows, y_true=y_true,
                                          X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                          w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

                if USE_SKEXTRA_FINAL and HAVE_SKEXTRA:
                    with track("final_kmedoids_fullD_skextra"):
                        km = SKKMedoids(n_clusters=K_FIXED, metric="precomputed", method="pam",
                                        init=np.array(init_meds_refined, int), max_iter=int(CFG.get("final_max_iter", 30)),
                                        random_state=0)
                        labels = km.fit_predict(D_full)
                else:
                    with track("final_kmedoids_fullD"):
                        algo_final = kmedoids(D_full, init_meds_refined, data_type="distance_matrix", ccore=True)
                        algo_final.process()
                        clusters_final = algo_final.get_clusters()
                    labels = np.empty(len(df), int)
                    for cid, idxs in enumerate(clusters_final): labels[idxs] = cid

            elif FINAL_MODE == "coreset":
                with track("build_coreset"):
                    base_core = (build_coreset_indices(global_cand_idx, MAX_CORESET) if USE_CAND_POOLS
                                 else np.array(init_meds_refined, int))
                    if CFG.get("enable_hard_reservoir", False) and len(global_hard_set) > 0:
                        hard = np.fromiter(global_hard_set, int)
                        core_idx = np.unique(np.concatenate([base_core, hard]))
                    else:
                        core_idx = base_core
                    if VERBOSE: print(f"[Coreset] size={core_idx.size} (max={MAX_CORESET})", flush=True)
                    if core_idx.size == 0:
                        raise RuntimeError("Coreset empty — enable candidate pools or increase cand_per_local.")
                    X_core = X_all[core_idx]

                if not time_left_ok() or (MIN_SECS_BEFORE_FINAL > 0 and secs_left() < MIN_SECS_BEFORE_FINAL):
                    medoid_rows = X_all[np.array(init_meds_refined, int)]
                    partial_save_and_exit("Not enough time left for coreset DM — seed-based assignment.",
                                          None, medoid_rows, core_idx=core_idx, y_true=y_true,
                                          X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                          w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

                with track("build_D_core"):
                    D_core = (pairwise_gower_block_w_with_progress if SHOW_PROGRESS or CORE_MEMMAP else pairwise_gower_block_w)(
                        X_core, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat,
                        desc=f"Build D_core (C={len(X_core)})",
                        use_memmap=CORE_MEMMAP,
                        memmap_path=str((outdir / "D_core.dat").resolve()) if CORE_MEMMAP else None
                    )

                if not time_left_ok() or (MIN_SECS_BEFORE_FINAL > 0 and secs_left() < MIN_SECS_BEFORE_FINAL):
                    medoid_rows = X_all[np.array(init_meds_refined, int)]
                    partial_save_and_exit("Not enough time left for coreset k-medoids — seed-based assignment.",
                                          None, medoid_rows, core_idx=core_idx, y_true=y_true,
                                          X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                          w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

                with track("final_kmedoids_coreset"):
                    core_set = set(core_idx.tolist())
                    init_in_core = [np.where(core_idx == m)[0][0] for m in init_meds_refined if m in core_set]
                    if len(init_in_core) < K_FIXED:
                        init_in_core = kmedoidspp_init_points_vec_w(X_core, K_FIXED, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat, seed=0)
                    algo_core = kmedoids(D_core, init_in_core, data_type="distance_matrix", ccore=True)
                    algo_core.process()
                    core_medoid_ids = algo_core.get_medoids()
                    medoid_rows = X_core[core_medoid_ids]

                if not time_left_ok():
                    labels = assign_by_nearest_medoid_blocked(X_all, medoid_rows, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                    partial_save_and_exit("Budget hit before final assignment — nearest-medoid labels saved.",
                                          labels, medoid_rows, core_idx=core_idx, y_true=y_true,
                                          X_all=X_all, IDX_NUM=IDX_NUM, IDX_BIN=IDX_BIN, IDX_CAT=IDX_CAT, NUM_RNG=NUM_RNG,
                                          w_num=w_num, w_bin=w_bin, w_cat=w_cat, used_per_feature=perw_used)

                with track("assign_nearest_medoid"):
                    labels = assign_by_nearest_medoid_blocked(X_all, medoid_rows, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
            else:
                raise ValueError(f"Unknown final_mode: {FINAL_MODE}")

        # -------- Metrics & logging --------
        with track("metrics"):
            if y_true is not None:
                ari = adjusted_rand_score(y_true, labels); nmi = normalized_mutual_info_score(y_true, labels)
            else:
                ari = nmi = np.nan
            if (FINAL_MODE == "fulld") and ("D_full" in locals()) and (SIL_SAMPLE is None):
                sil = silhouette_score(D_full, labels, metric="precomputed")
            else:
                idx = np.random.default_rng(1).choice(len(X_all), size=min(SIL_SAMPLE or 3000, len(X_all)), replace=False)
                X_s = X_all[idx]
                D_s = pairwise_gower_block_w(X_s, IDX_NUM, IDX_BIN, IDX_CAT, NUM_RNG, w_num, w_bin, w_cat)
                sil = silhouette_score(D_s, labels[idx], metric="precomputed")

            print(f"\nPipeline: streaming={USE_STREAMING}, weights={USE_WEIGHTS}, mode_w={CFG.get('weight_learning_mode','none')}, hungarian={USE_HUNGARIAN}, pools={USE_CAND_POOLS}, final={FINAL_MODE}, skextra_final={USE_SKEXTRA_FINAL}, progress={SHOW_PROGRESS}, memmap_fullD={FULLD_MEMMAP}, memmap_core={CORE_MEMMAP}", flush=True)
            print(f"ARI={float(ari):.3f}, NMI={float(nmi):.3f}, Sil={float(sil):.3f}", flush=True)

        sizes = pd.Series(labels).value_counts().sort_index()
        print("\nCluster sizes:", sizes.to_dict(), flush=True)
        if y_true is not None:
            summary = (pd.DataFrame({"pred": labels, "true": y_true})
                       .groupby("pred")["true"].agg(lambda s: s.value_counts().head(3).to_dict()))
            print("\nTop true labels per predicted cluster:")
            for k, top in summary.items(): print(f"  pred {k}: {top}", flush=True)

        with open(outdir / "config.json", "w") as f: json.dump(CFG, f, indent=2)
        total_secs = time.perf_counter() - START_TIME
        metrics = {"ARI": float(ari), "NMI": float(nmi), "Silhouette": float(sil),
                   "K": int(K_FIXED), "N": int(len(df)), "final_mode": FINAL_MODE,
                   "use_streaming": USE_STREAMING, "use_weights": USE_WEIGHTS,
                   "weight_learning_mode": CFG.get("weight_learning_mode","none"),
                   "use_hungarian": USE_HUNGARIAN, "use_candidate_pools": USE_CAND_POOLS,
                   "summary_weight": SUMMARY_W, "chunk_size": CHUNK_SIZE, "time_budget_secs": TIME_BUDGET_SECS,
                   "runtime_seconds": total_secs, "peak_rss_gb": PEAK_RSS_GB,
                   "use_sklearn_extra_final": USE_SKEXTRA_FINAL, "final_max_iter": FINAL_MAX_ITER,
                   "min_secs_before_final": MIN_SECS_BEFORE_FINAL}
        if FINAL_MODE == "coreset" and "core_idx" in locals():
            metrics["coreset_size"] = int(core_idx.size)
        # add top feature weights to metrics
        top_feat = sorted(perw_used.items(), key=lambda kv: kv[1], reverse=True)[:8]
        metrics["top_feature_weights"] = [{"feat": k, "w": float(v)} for k, v in top_feat]
        with open(outdir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
        pd.DataFrame(PROFILE).to_csv(outdir / "profile.csv", index=False)
        pd.Series(labels, name="label").to_csv(outdir / "labels.csv", index=False)
        if FINAL_MODE == "coreset" and "core_idx" in locals():
            np.save(outdir / "coreset_idx.npy", core_idx)
        # save full used weights
        with open(outdir / "learned_weights.json", "w") as f:
            json.dump({"per_feature_weights": perw_used, "learned_only": perw_learned}, f, indent=2)

        print(f"\n[Saved] run artifacts -> {outdir}", flush=True)

    finally:
        _stop_hb.set()
        try: threading.Thread.join(hb_thread, timeout=1)
        except Exception: pass
