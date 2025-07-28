# utils.py
import os
import pandas as pd
import streamlit as st
import numpy as np
import re
from scipy.signal.windows import hann, hamming, blackman

def load_clinical(f):
    """Read clinical table (latin1 fallback) → DataFrame."""
    try:
        if f.name.endswith('.csv'): return pd.read_csv(f, encoding='latin1')
        return pd.read_excel(f, engine='openpyxl')
    except Exception as e:
        st.error(f"Clinical file read error: {e}"); return None

def random_sphere_coords(n:int, seed:int=42):
    """Generate pseudo‑random but reproducible 3‑D coordinates on a sphere."""
    rng = np.random.RandomState(seed)
    phi = np.arccos(1 - 2*rng.rand(n))
    theta = 2*np.pi*rng.rand(n)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.vstack([x,y,z]).T

def compute_flexibility(dfc_stack, n_clusters=4):
    """
    Compute brain network flexibility for each ROI across dFC windows.
    Flexibility = fraction of times each ROI changes its cluster assignment.
    """
    from sklearn.cluster import KMeans
    if dfc_stack is None or len(dfc_stack) < 2:
        return None
    N = dfc_stack.shape[1]
    cluster_labels = []
    for m in dfc_stack:
        X = m
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        cluster_labels.append(labels)
    cluster_labels = np.array(cluster_labels)
    flexibility = []
    for roi in range(N):
        roi_labels = cluster_labels[:, roi]
        changes = np.sum(roi_labels[1:] != roi_labels[:-1])
        flexibility.append(changes / (len(roi_labels) - 1))
    return np.array(flexibility)

def find_significant_edges_continuous(
    data_df: pd.DataFrame,
    clinical_var: str = 'sara',
    feature_prefix: str = 'f',
    alpha: float = 0.05,
    method: str = 'fdr_bh',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Find top-n significant edges correlated with a clinical variable using Pearson correlation and FDR correction.
    Returns a DataFrame with edge, corr, pvalue, and qvalue columns.
    """
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    feature_cols = [c for c in data_df.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No columns start with '{feature_prefix}'")
    results = []
    y = data_df[clinical_var].values
    for edge in feature_cols:
        x = data_df[edge].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 5:
            continue
        corr, pval = pearsonr(x[mask], y[mask])
        results.append({'edge': edge, 'corr': corr, 'pvalue': pval})
    res_df = pd.DataFrame(results)
    if res_df.empty:
        raise RuntimeError("No valid feature-clinical pairs to test.")
    res_df['qvalue'] = multipletests(res_df['pvalue'].values, alpha=alpha, method=method)[1]
    res_df = res_df.sort_values('qvalue').reset_index(drop=True)
    return res_df.head(top_n)


def make_windows(ts, length, step, wtype=None):
    """
    Create overlapping windows from a time-series array with optional windowing function.
    ts: array (T, N)
    length: window length
    step: window step
    wtype: 'hann', 'hamming', 'blackman', or None
    Returns: array of shape (num_windows, length, N)
    """
    T, R = ts.shape
    n = 1 + (T - length) // step
    out = np.empty((n, length, R), dtype=np.float32)
    w = None
    if wtype == "hann":
        w = hann(length, sym=False).astype(np.float32)[:, None]
    elif wtype == "hamming":
        w = hamming(length, sym=False).astype(np.float32)[:, None]
    elif wtype == "blackman":
        w = blackman(length, sym=False).astype(np.float32)[:, None]
    for i in range(n):
        s = i * step
        seg = ts[s: s + length].copy()
        if w is not None:
            seg *= w
        out[i] = seg
    return out

def cov_dist(a, b):
    return np.linalg.norm(a - b, 'fro') / np.linalg.norm(a, 'fro')

def conn_metric(windows):
    corrs = [np.corrcoef(w.T) for w in windows]
    corr_std = np.std(corrs, axis=0)
    return np.mean(corr_std)

def compute_network_measures(fc_matrix):
    """
    Compute degree centrality, global efficiency, clustering coefficient, and modularity for a given FC matrix.
    Returns: dc, ge, cc, mod
    """
    import networkx as nx
    import community as community_louvain
    G = nx.from_numpy_array(fc_matrix)
    dc   = np.mean(list(nx.degree_centrality(G).values()))
    ge   = nx.global_efficiency(G)
    cc   = nx.average_clustering(G, weight='weight')
    part = community_louvain.best_partition(G, weight='weight')
    mod  = community_louvain.modularity(part, G, weight='weight')
    return dc, ge, cc, mod


def extract_upper_triangle(mat: np.ndarray) -> np.ndarray:
    """
    Extract the upper triangle (excluding diagonal) of a square matrix as a 1D vector.
    """
    triu = np.triu_indices_from(mat, k=1)
    return mat[triu]

# You can add more utility functions here as needed for the project

from utils import make_windows, cov_dist, conn_metric, compute_network_measures, extract_upper_triangle

# نمونه استفاده عملی از توابع utils در انتهای فایل
if __name__ == "__main__":
    # داده تست تصادفی
    ts = np.random.randn(200, 10)  # 200 timepoints, 10 ROIs
    windows = make_windows(ts, length=60, step=30, wtype='hann')
    print("windows shape:", windows.shape)
    dyn_metric = conn_metric(windows)
    print("Dynamic connectivity metric:", dyn_metric)
    full_cov = np.cov(ts.T, dtype=np.float64)
    covs = np.mean([np.cov(w.T, dtype=np.float64) for w in windows], axis=0)
    cov_loss = cov_dist(full_cov, covs)
    print("Covariance loss:", cov_loss)
    dc, ge, cc, mod = compute_network_measures(full_cov)
    print(f"Degree Centrality: {dc}, Global Efficiency: {ge}, Clustering Coefficient: {cc}, Modularity: {mod}")
    upper_vec = extract_upper_triangle(full_cov)
    print("Upper triangle vector length:", len(upper_vec))
