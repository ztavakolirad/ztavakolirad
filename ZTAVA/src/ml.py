# ml.py
import pandas as pd
import streamlit as st
from config import opt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from utils import make_windows, cov_dist, conn_metric, compute_network_measures, extract_upper_triangle
import numpy as np

def svm_classifier(X, y, groups, C=1.0):
    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, kernel='rbf', random_state=42))
    ])
    gkf = GroupKFold(n_splits=3)
    acc = cross_val_score(pipe, X, y, cv=gkf, groups=groups).mean()
    return acc

def rf_regressor(X, y, groups, n_tree=100):
    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=n_tree, random_state=42))
    ])
    gkf = GroupKFold(n_splits=3)
    mse = -cross_val_score(pipe, X, y, cv=gkf, groups=groups,
                        scoring='neg_mean_squared_error').mean()
    return mse

def run_ml_models(X_raw, X_resid, y, groups, model_func):
    """Run the same ML model on raw and residual FC, return metrics for both."""
    res = {}
    res['raw'] = model_func(X_raw, y, groups)
    res['residual'] = model_func(X_resid, y, groups)
    return res

# Example usage in ml.py

def example_feature_extraction(ts):
    # ساخت پنجره‌های زمانی
    windows = make_windows(ts, length=60, step=30, wtype='hann')
    # محاسبه پویایی همبستگی
    dyn_metric = conn_metric(windows)
    # محاسبه کوواریانس کلی و میانگین پنجره‌ها
    full_cov = np.cov(ts.T, dtype=np.float64)
    covs = np.mean([np.cov(w.T, dtype=np.float64) for w in windows], axis=0)
    cov_loss = cov_dist(full_cov, covs)
    # محاسبه معیارهای شبکه
    dc, ge, cc, mod = compute_network_measures(full_cov)
    # استخراج بخش بالایی ماتریس
    upper_vec = extract_upper_triangle(full_cov)
    return {
        'dyn_metric': dyn_metric,
        'cov_loss': cov_loss,
        'dc': dc,
        'ge': ge,
        'cc': cc,
        'mod': mod,
        'upper_vec': upper_vec
    }
