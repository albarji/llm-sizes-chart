"""Utility functions for trend analysis of LLM sizes."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def compute_trend(dates, sizes):
    """Computes the trend line for LLM sizes over time using SVR."""
    X = dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.log10(sizes.astype(float).values)
    svr = SVR(C=1e1).fit(X, y)

    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_fit = svr.predict(X_fit)
    return pd.to_datetime(scaler.inverse_transform(X_fit).flatten(), unit="s"), 10**y_fit
