"""
Interactive Plotly visualizations for Qlimate.

Used for:
  - Climate data exploration (choropleth maps, time series)
  - Model comparison dashboard
  - Quantum vs classical scatter
  - 3D PCA visualization
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "Normal": "#4CAF50",
    "Drought": "#FF9800",
    "Wet_Flood": "#2196F3",
    "Heat_Extreme": "#F44336",
    "Cold_Extreme": "#9C27B0",
}

MODEL_COLORS = {
    "Random Forest": "#1565C0",
    "SVM (RBF)": "#1976D2",
    "XGBoost": "#42A5F5",
    "Neural Network": "#90CAF9",
    "QSVC": "#6A1B9A",
    "VQC": "#AB47BC",
}


def india_climate_explorer(
    df: pd.DataFrame,
    geojson: dict,
    variable: str = "T2M",
    title: Optional[str] = None,
) -> go.Figure:
    """Interactive India choropleth with year slider and variable dropdown.

    Args:
        df: state-level monthly DataFrame
        geojson: India states GeoJSON dict
        variable: default variable to display
    """
    df = df.copy()
    # Convert temperature to Celsius
    if variable == "T2M" and df[variable].max() > 200:
        df["T2M_display"] = df["T2M"] - 273.15
        display_var = "T2M_display"
        label = "Temperature (°C)"
    elif variable == "PRECTOT":
        df["PRECTOT_display"] = df["PRECTOT"] * 86400 * 30  # mm/month
        display_var = "PRECTOT_display"
        label = "Precipitation (mm/month)"
    else:
        display_var = variable
        label = variable

    annual = df.groupby(["year", "state"])[display_var].mean().reset_index()

    fig = px.choropleth(
        annual,
        geojson=geojson,
        locations="state",
        featureidkey="properties.state",
        color=display_var,
        animation_frame="year",
        color_continuous_scale="RdYlBu_r" if "T2M" in display_var else "YlGnBu",
        title=title or f"India Climate: {label} by State (1995–2024)",
        labels={display_var: label},
        hover_name="state",
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcoastlines=True,
        coastlinecolor="gray",
        showframe=False,
    )
    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(len=0.7),
    )
    return fig


def model_comparison_dashboard(all_results: dict) -> go.Figure:
    """Multi-panel dashboard: radar chart + bar chart + confusion matrices."""
    models = list(all_results.keys())
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Performance Metrics", "Training Time (seconds)"],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    for metric, label in zip(metrics, metric_labels):
        vals = [all_results[m].get(metric, 0) for m in models]
        fig.add_trace(
            go.Bar(name=label, x=models, y=vals, text=[f"{v:.3f}" for v in vals],
                   textposition="outside"),
            row=1, col=1,
        )

    train_times = [all_results[m].get("training_time", 0) for m in models]
    fig.add_trace(
        go.Bar(
            name="Training Time",
            x=models,
            y=train_times,
            text=[f"{t:.1f}s" for t in train_times],
            textposition="outside",
            marker_color=[MODEL_COLORS.get(m, "#888888") for m in models],
        ),
        row=1, col=2,
    )

    fig.update_yaxes(range=[0, 1.15], row=1, col=1)
    fig.update_layout(
        title="Classical vs Quantum ML — Full Comparison",
        height=500,
        barmode="group",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def quantum_vs_classical_scatter(all_results: dict) -> go.Figure:
    """Scatter plot: training time (x, log) vs F1 score (y).

    Point size proportional to number of model parameters (if available).
    Color by model type (classical = blue, quantum = purple).
    """
    records = []
    for name, r in all_results.items():
        records.append({
            "model": name,
            "f1_macro": r.get("f1_macro", 0),
            "training_time": max(r.get("training_time", 0.1), 0.1),
            "accuracy": r.get("accuracy", 0),
            "type": "Quantum" if name in ("QSVC", "VQC") else "Classical",
        })
    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x="training_time",
        y="f1_macro",
        color="type",
        color_discrete_map={"Classical": "#1565C0", "Quantum": "#6A1B9A"},
        text="model",
        size="accuracy",
        size_max=30,
        log_x=True,
        title="Training Time vs F1 Macro — Classical vs Quantum",
        labels={
            "training_time": "Training Time (s, log scale)",
            "f1_macro": "F1 Macro",
            "type": "Model Type",
        },
        hover_data={"accuracy": ":.3f", "f1_macro": ":.3f"},
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1.5,
                                                                        color="white")))
    fig.update_layout(
        height=480,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray", range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def pca_3d_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    title: str = "PCA-Reduced Features — Quantum Training Data",
) -> go.Figure:
    """Interactive 3D scatter of first 3 PCA components, colored by class."""
    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2] if X_pca.shape[1] > 2 else np.zeros(len(y)),
        "class": [class_names[yi] if yi < len(class_names) else str(yi) for yi in y],
    })

    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3",
        color="class",
        color_discrete_map=CLASS_COLORS,
        title=title,
        opacity=0.75,
        hover_data={"class": True},
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        height=580,
        legend_title="Climate Condition",
        scene=dict(
            xaxis_title="PC1 (angle-encoded)",
            yaxis_title="PC2 (angle-encoded)",
            zaxis_title="PC3 (angle-encoded)",
        ),
    )
    return fig


def climate_condition_time_series(
    df: pd.DataFrame,
    states: Optional[list[str]] = None,
    class_names: Optional[list[str]] = None,
) -> go.Figure:
    """Stacked area of climate condition frequency over time (all states or selected)."""
    if states:
        df = df[df["state"].isin(states)]
    if class_names is None:
        class_names = ["Normal", "Drought", "Wet_Flood", "Heat_Extreme", "Cold_Extreme"]

    # Fraction of each class per year
    yearly = (
        df.groupby(["year", "label_name"]).size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum(), axis=1)
        * 100
    )
    # Ensure all classes present
    for cls in class_names:
        if cls not in yearly.columns:
            yearly[cls] = 0.0
    yearly = yearly[class_names]

    fig = go.Figure()
    for cond in reversed(class_names):
        if cond in yearly.columns:
            fig.add_trace(go.Scatter(
                x=yearly.index,
                y=yearly[cond],
                name=cond,
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=CLASS_COLORS.get(cond, "gray")),
                fillcolor=CLASS_COLORS.get(cond, "gray"),
                hovertemplate=f"{cond}: %{{y:.1f}}%<extra></extra>",
            ))

    title_suffix = f" ({', '.join(states)})" if states else " (All India)"
    fig.update_layout(
        title=f"Climate Condition Frequency Over Time{title_suffix}",
        xaxis_title="Year",
        yaxis_title="% of state-month records",
        yaxis=dict(range=[0, 100]),
        height=450,
        legend=dict(orientation="h", y=-0.15, traceorder="reversed"),
        hovermode="x unified",
    )
    return fig


def quantum_circuit_stats_table(
    circuit_info: dict,
) -> go.Figure:
    """Table comparing QSVC vs VQC circuit properties."""
    headers = ["Property", "QSVC", "VQC"]
    qsvc = circuit_info.get("qsvc", {})
    vqc = circuit_info.get("vqc", {})

    rows = [
        ["Qubits", qsvc.get("n_qubits", 4), vqc.get("n_qubits", 4)],
        ["Circuit Depth", qsvc.get("circuit_depth", "—"), vqc.get("circuit_depth", "—")],
        ["Parameters", "None (kernel)", vqc.get("n_params", "—")],
        ["Training Samples", qsvc.get("n_train", 400), vqc.get("n_train", 400)],
        ["Optimizer", "N/A", "COBYLA"],
        ["Feature Map", "ZZFeatureMap (reps=1)", "ZZFeatureMap (reps=1)"],
        ["Ansatz", "N/A", "EfficientSU2 (reps=2)"],
        ["Entanglement", "Linear", "Linear"],
        ["Training Time (s)", f"{qsvc.get('training_time', '—'):.1f}",
         f"{vqc.get('training_time', '—'):.1f}"],
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color="#6A1B9A",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[[r[0] for r in rows],
                    [r[1] for r in rows],
                    [r[2] for r in rows]],
            fill_color=[["#F3E5F5" if i % 2 == 0 else "white"
                         for i in range(len(rows))]],
            align="left",
            font=dict(size=11),
        ),
    ))
    fig.update_layout(
        title="Quantum Circuit Comparison: QSVC vs VQC",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def save_html_dashboard(
    figures: list[go.Figure],
    titles: list[str],
    output_path: Path,
) -> None:
    """Combine multiple Plotly figures into a single self-contained HTML file."""
    import plotly.io as pio

    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Qlimate — Classical vs Quantum ML Dashboard</title>",
        "<style>",
        "body { font-family: 'Georgia', serif; background: #0d1117; color: #e6edf3; "
        "margin: 0; padding: 20px; }",
        "h1 { text-align: center; color: #a78bfa; font-size: 2em; margin-bottom: 10px; }",
        "h2 { color: #7c3aed; border-bottom: 1px solid #333; padding-bottom: 8px; }",
        ".section { background: #161b22; border-radius: 8px; padding: 20px; "
        "margin-bottom: 30px; }",
        "</style>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "</head><body>",
        "<h1>Qlimate: Classical ML vs Quantum ML on NASA MERRA-2 Climate Data</h1>",
        "<p style='text-align:center;color:#888'>",
        "India State-Level Climate Condition Classification | 1995–2024 | 4 Qubits",
        "</p>",
    ]

    for title, fig in zip(titles, figures):
        div_html = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                               div_id=f"fig_{title.replace(' ', '_')}")
        html_parts.append(f"<div class='section'><h2>{title}</h2>{div_html}</div>")

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info(f"Saved HTML dashboard: {output_path}")
