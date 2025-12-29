import colorcet as cc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = pd.read_csv("llm_sizes.csv")

df["Date"] = pd.to_datetime(df["Date"])

cats = list(df["Creator"].astype(str).unique())

colors = cc.glasbey[:7]
symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "triangle-left", "triangle-right", "star", "hexagon"]  # 11 symbols
n_colors = len(colors)
n_symbols = len(symbols)

# Build color and symbol maps
color_map = {c: colors[i % n_colors] for i, c in enumerate(cats)}
symbol_map = {c: symbols[(i // n_colors) % n_symbols] for i, c in enumerate(cats)}

# Data points with alternate text positions to avoid overlap
alternate_text_positions = {
    "T5-11B": "top center",
    "Turing-NLG": "top center",
    "BLOOM": "top center",
    "Salamandra": "middle left",
    "RigoChat 2": "middle right",
    "PaLM": "top center",
    "Qwen3-Max": "top center",
}

fig = go.Figure()

for creator in cats:
    group = df[df["Creator"] == creator]
    textpositions = [alternate_text_positions.get(llm, "bottom center") for llm in group["LLM"]]
    fig.add_trace(go.Scatter(
        x=group["Date"],
        y=group["Size (B)"],
        mode="markers+text",
        text=group["LLM"],
        textposition=textpositions,
        marker=dict(
            size=16,
            color=color_map[creator],
            symbol=symbol_map[creator]
        ),
        name=creator
    ))

fig.update_layout(
    title={"text": "Large Language Model (LLM) Sizes Over Time (2018-2025)", "font": dict(size=28)},
    xaxis_title={"text": "Publication date", "font": dict(size=22)},
    yaxis_title={"text": "Number of parameters (billions, log scale)", "font": dict(size=22)},
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.21, xanchor="center", x=0.5),
    yaxis_type="log",
    xaxis=dict(tickfont=dict(size=18)),
    yaxis=dict(tickfont=dict(size=18))
)

# Add trend line
X = df["Date"].map(pd.Timestamp.timestamp).values.reshape(-1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.log10(df["Size (B)"].astype(float).values)
svr = SVR(C=1e2).fit(X, y)

X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_fit = svr.predict(X_fit)

fig.add_trace(go.Scatter(
    x=pd.to_datetime(scaler.inverse_transform(X_fit).flatten(), unit="s"),
    y=10**y_fit,
    mode="lines",
    line=dict(color="black", dash="dash"),
    name="Trend"
))

fig.write_image("llm_sizes_chart.png", width=1600, height=900)