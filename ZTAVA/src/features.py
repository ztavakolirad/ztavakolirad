# features.py
import os, warnings
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
from config import opt

@st.cache_data(show_spinner="📊 Calculation of static characteristics...")
def calc_static_feats(ts_path:str, feats:list):
    """
    Calculates static graph metrics from an ROI time‑series CSV.
    Returns dict or None on failure.
    """
    if not os.path.exists(ts_path): return None
    try:
        ts = pd.read_csv(ts_path)
        if ts.shape[1] < 2: return None
        conn = ts.corr().fillna(0).values
        G = nx.from_numpy_array(np.abs(conn))
        res = {}
        if 'DegC' in feats:
            res['avg_deg_cent'] = np.mean(list(nx.degree_centrality(G).values()))
        if 'Clust' in feats:
            res['avg_clust'] = nx.average_clustering(G)
        if opt['bct']:
            if 'Mods' in feats:
                _, q = bct.modularity_louvain_und(np.abs(conn))
                res['modularity'] = q
            if 'GE' in feats:
                res['glob_eff'] = bct.efficiency_wei(np.abs(conn))
        # ----- Community assignments for later 3D rendering ---------- #
        if 'community_vec' in feats:
            if opt['bct']:
                comm, _ = bct.modularity_louvain_und(np.abs(conn))
            else:
                # fallback: greedy modularity via networkx
                comm_iter = nx.algorithms.community.greedy_modularity_communities(G)
                comm = np.zeros(len(G), dtype=int)
                for cidx, nodes in enumerate(comm_iter):
                    for n in nodes: comm[n] = cidx
            res['community_vec'] = comm
        return res
    except Exception as e:
        warnings.warn(str(e))
        return None

@st.cache_data(show_spinner="⚡ Dynamic FC calculation...")
def calc_dfc(ts_path:str, win:int, step:int):
    """
    Sliding‑window correlation → variance metric + full dFC stack.
    Returns dict with variance scalar & 3‑D array of matrices.
    """
    if not os.path.exists(ts_path): return None
    try:
        ts = pd.read_csv(ts_path)
        T, N = ts.shape   # timepoints × ROIs
        if T < win: return None
        mats = []
        for s in range(0, T-win+1, step):
            window = ts.iloc[s:s+win]
            mats.append(window.corr().fillna(0).values)
        dFC = np.array(mats)
        var_conn = np.var(dFC, axis=0)
        mean_var = np.mean(var_conn[np.triu_indices(N, 1)])
        return {'dFC_variance': mean_var, 'dFC_stack': dFC}
    except Exception as e:
        warnings.warn(str(e)); return None

def compute_plv(ts:pd.DataFrame, fs:int=1):
    """
    Very lightweight Phase‑Locking Value (PLV) estimation.
    Steps: Hilbert Φ extraction ➜ PLV across ROI pairs ➜ mean value.
    """
    if not opt['pywt']:
        st.warning("PyWavelets not installed ➜ Phase synchrony is disabled"); return None
    from scipy.signal import hilbert
    phases = np.angle(hilbert(ts.values, axis=0))
    N = phases.shape[1]
    plv_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            diff = phases[:,i]-phases[:,j]
            plv = abs(np.mean(np.exp(1j*diff)))
            plv_mat[i,j] = plv_mat[j,i] = plv
    return plv_mat
