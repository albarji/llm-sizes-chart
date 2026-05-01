import colorcet as cc
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from trend import compute_trend

df = pd.read_csv("llm_sizes.csv")

df["Date"] = pd.to_datetime(df["Date"])

colors = cc.glasbey[:7]
symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "triangle-left", "triangle-right", "star", "hexagon"]  # 11 symbols
n_colors = len(colors)
n_symbols = len(symbols)

# Build color and symbol maps
creators = list(df["Creator"].astype(str).unique())
color_map = {c: colors[i % n_colors] for i, c in enumerate(creators)}
symbol_map = {c: symbols[(i // n_colors) % n_symbols] for i, c in enumerate(creators)}

# Data points with alternate text positions to avoid overlap
alternate_text_positions = {
    "T5-11B": "middle right",
    "Turing-NLG": "middle right",
    "Salamandra": "middle left",
    "RigoChat 2": "middle right",
    "PaLM": "top center",
    "Qwen3-Max": "top center",
    "Megatron-LM": "middle left",
    "OPT-175B": "middle left",
    "Qwen2-72B": "top center",
    "RigoBERTa 2": "top center",
    "MarIA": "top center",
    "BERT Large": "middle left",
    "RoBERTa-XLM": "top center",
    "GPT-2": "middle left",
    "Megatron-Turing NLG": "middle left",
    "gpt-oss-120B": "middle left",
    "LLaMA 2 70B": "bottom left",
    "DeepSeek-R1": "top center",
    "DeepSeek-3.2": "middle left",
    "Llama 4 Maverick": "middle left",
    "Qwen3-235B-A22B": "middle left",
    "GPT-3": "top center",
    "Llama 3 405B": "middle left",
    "Qwen3.5-397B-A17B": "bottom center",
    "GLM-5.1": "middle right",
    "DeepSeek-V4-Pro": "top center",
}

fig = go.Figure()

# Compute and plot trend lines for Decoders and Encoders
decoders = df[df["Architecture"] == "Decoder"]
x_trend_decoders, y_trend_decoders = compute_trend(decoders["Date"], decoders["Size (B)"])

fig.add_trace(
    go.Scatter(
        x=x_trend_decoders,
        y=y_trend_decoders,
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Decoders trend"
    )
)

encoders = df[df["Architecture"] == "Encoder"]
x_trend_encoders, y_trend_encoders = compute_trend(encoders["Date"], encoders["Size (B)"])

fig.add_trace(
    go.Scatter(
        x=x_trend_encoders,
        y=y_trend_encoders,
        mode="lines",
        line=dict(color="gray", dash="dot"),
        name="Encoders trend"
    )
)

# Plot data points

for creator in creators:
    group = df[(df["Creator"] == creator) & (df["Architecture"] == "Decoder")]
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
            symbol=symbol_map[creator],
        ),
        name=creator
    ))
    group = df[(df["Creator"] == creator) & (df["Architecture"] == "Encoder")]
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
            symbol=symbol_map[creator],
            line={"width": 4, "color": 'DarkSlateGrey'}
        ),
        name=creator
    ))

fig.update_layout(
    title={"text": "Large Language Model (LLM) Sizes Over Time (2018-2026)", "font": dict(size=28)},
    xaxis_title={"text": "Publication date", "font": dict(size=22)},
    yaxis_title={"text": "Number of parameters (billions, log scale)", "font": dict(size=22)},
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.4),
    yaxis_type="log",
    xaxis=dict(tickfont=dict(size=18)),
    yaxis=dict(tickfont=dict(size=18)),
)

fig.add_layout_image(
    dict(
        source=Image.open("assets/cc-by-4.0.png"),
        x=1,
        y=-0.25,
        xanchor="right",
        yanchor="bottom",
        sizex=0.11,
        sizey=0.11,
        sizing="contain",
        layer="above"
    )
)

fig.add_annotation(
    xref="paper",
    yref="paper",
    x=1,
    y=-0.155,
    xanchor="right",
    yanchor="bottom",
    text="Álvaro Barbero Jiménez",
    showarrow=False,
    font=dict(size=13, color="black"),
    align="right"
)

fig.write_image("llm_sizes_chart.png", width=1600, height=900)