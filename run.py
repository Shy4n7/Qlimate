"""
Qlimate — full pipeline runner.

Stages:
  1. preprocess  — aggregate MERRA-2 grid to India states
  2. label       — assign climate condition labels
  3. engineer    — feature engineering + train/val/test splits
  4. classical   — train RF, SVM, XGBoost, Neural Network
  5. quantum     — train QSVC and VQC
  6. evaluate    — unified comparison report
  7. visualize   — static figures + interactive dashboard

Usage:
  python run.py                    # full pipeline
  python run.py --from preprocess  # start from a specific stage
  python run.py --only classical   # run one stage only
  python run.py --skip quantum     # skip a stage
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STAGES = ["preprocess", "label", "engineer", "classical", "quantum", "evaluate", "visualize",
          "export_metrics", "visualize_extended"]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_preprocess(config: dict) -> None:
    from src.data.preprocess import process_all
    logger.info("=== STAGE: preprocess ===")
    df = process_all(config)
    logger.info(f"Preprocessed dataset: {df.shape[0]} rows, {df.shape[1]} columns")


def stage_label(config: dict) -> None:
    from src.data.label import label_dataset
    logger.info("=== STAGE: label ===")
    df = label_dataset(config)
    logger.info(f"Labeled dataset: {df.shape[0]} rows")
    dist = df["label_name"].value_counts()
    for name, count in dist.items():
        logger.info(f"  {name}: {count} ({100*count/len(df):.1f}%)")


def stage_engineer(config: dict) -> None:
    import pandas as pd
    from src.features.engineering import prepare_splits, prepare_quantum_subset, save_artifacts
    logger.info("=== STAGE: engineer ===")

    processed_dir = Path(config["paths"]["processed_data"])
    labeled_path = processed_dir / "merra2_india_labeled.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Run 'label' stage first: {labeled_path}")

    df = pd.read_csv(labeled_path)
    splits = prepare_splits(df, config)
    quantum = prepare_quantum_subset(
        splits["X_train"], splits["y_train"],
        splits["X_test"], splits["y_test"],
        n_components=config["quantum_ml"]["pca_components"],
        subset_size=config["quantum_ml"]["training_subset_size"],
        random_state=config["data_split"]["random_state"],
    )
    save_artifacts(splits, quantum, config)
    logger.info(
        f"Splits — train: {splits['X_train'].shape[0]}, "
        f"val: {splits['X_val'].shape[0]}, test: {splits['X_test'].shape[0]}"
    )
    logger.info(
        f"Quantum subset: {quantum['X_train_q'].shape[0]} samples, "
        f"PCA explains {quantum['pca_explained_variance']:.1%}"
    )


def stage_classical(config: dict) -> dict:
    from src.features.engineering import load_splits
    from src.models.classical import ClassicalModelTrainer
    from src.evaluation.metrics import evaluate_model, compare_models
    logger.info("=== STAGE: classical ===")

    splits, _ = load_splits(config)
    trainer = ClassicalModelTrainer(config)

    results = trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )

    models_dir = Path(config["paths"]["models"])
    trainer.save_models(models_dir)

    # Evaluate and print comparison
    import torch

    def make_predict_fn(name: str, res: dict, trainer: "ClassicalModelTrainer"):
        if name == "neural_network":
            model = res["model"]
            device = trainer.device

            def nn_predict(X: np.ndarray) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    t = torch.tensor(X, dtype=torch.float32).to(device)
                    return model(t).argmax(dim=1).cpu().numpy()

            return nn_predict
        return lambda X, m=res["model"]: m.predict(X)

    eval_results = {}
    for name, res in results.items():
        metrics = evaluate_model(
            make_predict_fn(name, res, trainer),
            splits["X_test"], splits["y_test"], name
        )
        eval_results[name] = metrics
        logger.info(
            f"  {name:20s}  acc={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1_macro']:.3f}  time={res['training_time']:.1f}s"
        )

    return eval_results


def stage_quantum(config: dict) -> dict:
    from src.features.engineering import load_splits
    from src.models.quantum import QuantumModelTrainer
    from src.evaluation.metrics import evaluate_model
    logger.info("=== STAGE: quantum ===")

    splits, quantum = load_splits(config)
    trainer = QuantumModelTrainer(config)

    figures_dir = Path(config["paths"]["figures"])
    trainer.draw_circuits(figures_dir)

    models_dir = Path(config["paths"]["models"])

    qsvc_result = trainer.train_qsvc(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
    )
    trainer.save_models(models_dir)  # save after QSVC in case VQC crashes

    vqc_result = trainer.train_vqc(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
    )
    trainer.save_models(models_dir)  # save again with VQC included

    eval_results = {}
    for name, res in [("QSVC", qsvc_result), ("VQC", vqc_result)]:
        metrics = evaluate_model(
            lambda X, r=res: r["preds"],  # predictions already computed
            quantum["X_test_q"], quantum["y_test_q"], name,
            precomputed_preds=res["preds"],
        )
        eval_results[name] = metrics
        logger.info(
            f"  {name:20s}  acc={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1_macro']:.3f}  time={res['training_time']:.1f}s"
        )

    return eval_results


def load_eval_results(config: dict) -> tuple[dict, dict]:
    """Re-evaluate saved classical and quantum models on test data."""
    import joblib
    import torch
    from src.features.engineering import load_splits
    from src.models.classical import ClimateNN
    from src.evaluation.metrics import evaluate_model
    from qiskit_machine_learning.algorithms import QSVC

    splits, quantum = load_splits(config)
    models_dir = Path(config["paths"]["models"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classical_results: dict = {}
    for name, fname in [
        ("random_forest", "random_forest.pkl"),
        ("svm", "svm.pkl"),
        ("xgboost", "xgboost.pkl"),
    ]:
        path = models_dir / fname
        if not path.exists():
            continue
        model = joblib.load(path)
        metrics = evaluate_model(
            lambda X, m=model: m.predict(X),
            splits["X_test"], splits["y_test"], name
        )
        classical_results[name] = metrics

    nn_path = models_dir / "neural_network_state_dict.pt"
    if nn_path.exists():
        meta = joblib.load(models_dir / "neural_network_meta.pkl")
        n_features = splits["X_test"].shape[1]
        cfg_nn = config["classical_ml"]["neural_net"]
        n_classes = len(np.unique(splits["y_test"]))
        model = ClimateNN(n_features, cfg_nn["hidden_dims"], n_classes, cfg_nn["dropout"]).to(device)
        model.load_state_dict(torch.load(nn_path, map_location=device))
        model.eval()

        def nn_predict(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32).to(device)
                return model(t).argmax(dim=1).cpu().numpy()

        metrics = evaluate_model(nn_predict, splits["X_test"], splits["y_test"], "neural_network")
        classical_results["neural_network"] = metrics

    quantum_results: dict = {}
    for qname in ("qsvc", "vqc"):
        meta_path = models_dir / f"{qname}_meta.pkl"
        if not meta_path.exists():
            continue
        meta = joblib.load(meta_path)
        preds = meta.get("preds")
        if preds is not None:
            metrics = evaluate_model(None, quantum["X_test_q"], quantum["y_test_q"],
                                      qname.upper(), precomputed_preds=preds)
            metrics["training_time"] = meta.get("training_time", 0)
            metrics["loss_history"] = meta.get("loss_history", [])
            quantum_results[qname.upper()] = metrics

    return classical_results, quantum_results


def stage_evaluate(config: dict, classical_results: dict, quantum_results: dict) -> None:
    from src.evaluation.metrics import compare_models
    logger.info("=== STAGE: evaluate ===")

    all_results = {**classical_results, **quantum_results}
    comparison = compare_models(all_results)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    logger.info("\nModel comparison (sorted by F1):")
    logger.info(comparison.to_string(index=False))


def stage_visualize(config: dict, classical_results: dict, quantum_results: dict) -> None:
    from src.features.engineering import load_splits
    from src.visualization.static_plots import (
        plot_model_comparison_bar,
        plot_confusion_matrices,
        plot_scalability,
        create_summary_figure,
    )
    from src.visualization.interactive import save_html_dashboard
    from src.evaluation.metrics import compare_models
    logger.info("=== STAGE: visualize ===")

    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = {**classical_results, **quantum_results}
    comparison = compare_models(all_results)

    splits, quantum = load_splits(config)

    plot_model_comparison_bar(comparison, figures_dir / "model_comparison.png")
    plot_confusion_matrices(all_results, figures_dir / "confusion_matrices.png")
    vqc_loss = quantum_results.get("VQC", {}).get("loss_history", [])
    create_summary_figure(comparison, all_results, vqc_loss, figures_dir / "summary.png")

    # Interactive dashboard
    from src.visualization.interactive import (
        model_comparison_dashboard,
        quantum_vs_classical_scatter,
    )
    figs = [
        model_comparison_dashboard(all_results),
        quantum_vs_classical_scatter(all_results),
    ]
    titles = ["Model Comparison", "Quantum vs Classical"]
    save_html_dashboard(figs, titles, Path("results/dashboard.html"))

    logger.info(f"Figures saved to {figures_dir}")
    logger.info("Interactive dashboard saved to results/dashboard.html")


def stage_export_metrics(config: dict) -> None:
    logger.info("=== STAGE: export_metrics ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/export_metrics.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("export_metrics.py failed")
    else:
        logger.info("Metrics exported to results/metrics/")


def stage_visualize_extended(config: dict) -> None:
    import joblib
    from src.features.engineering import load_splits
    from src.visualization.optimization_plots import (
        plot_nn_loss_curves, plot_vqc_no_convergence, plot_combined_optimization,
    )
    from src.visualization.scaling_plots import (
        plot_quantum_scaling, plot_classical_vs_quantum_scaling,
    )
    from src.visualization.feature_flow import plot_feature_compression_diagram
    from src.visualization.pipeline_breakdown import plot_pipeline_timing
    from src.visualization.kernel_visualization import (
        kernel_matrix_from_model, plot_kernel_matrix_heatmap,
    )

    logger.info("=== STAGE: visualize_extended ===")
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, quantum = load_splits(config)
    pca = joblib.load(Path(config["paths"]["models"]) / "pca_model.pkl")
    nn_meta = joblib.load(Path(config["paths"]["models"]) / "neural_network_meta.pkl")
    vqc_meta = joblib.load(Path(config["paths"]["models"]) / "vqc_meta.pkl")
    vqc_meta["maxiter_configured"] = config["quantum_ml"]["vqc"]["maxiter"]

    plot_nn_loss_curves(nn_meta["train_losses"], nn_meta["val_losses"],
                        figures_dir / "nn_loss_curves.png")
    plot_vqc_no_convergence(vqc_meta, figures_dir / "vqc_no_convergence.png")
    plot_combined_optimization(nn_meta, vqc_meta, figures_dir / "optimization_comparison.png")
    plot_quantum_scaling(figures_dir / "quantum_scaling.png")
    plot_classical_vs_quantum_scaling(figures_dir / "scaling_comparison.png")
    plot_feature_compression_diagram(pca.explained_variance_ratio_.tolist(),
                                     figures_dir / "feature_compression.png")
    plot_pipeline_timing(figures_dir / "pipeline_breakdown.png")

    logger.info("Computing kernel matrix (may take ~30s)...")
    K = kernel_matrix_from_model(quantum["X_train_q"], n_vis=50, config=config)
    plot_kernel_matrix_heatmap(K, figures_dir / "kernel_matrix.png")

    logger.info(f"Extended figures saved to {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qlimate pipeline runner")
    parser.add_argument(
        "--from", dest="from_stage", choices=STAGES, default=None,
        help="Start pipeline from this stage (skips earlier stages)"
    )
    parser.add_argument(
        "--only", choices=STAGES, default=None,
        help="Run only this stage"
    )
    parser.add_argument(
        "--skip", choices=STAGES, default=None,
        help="Skip this stage"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Determine which stages to run
    if args.only:
        active_stages = [args.only]
    elif args.from_stage:
        start = STAGES.index(args.from_stage)
        active_stages = STAGES[start:]
    else:
        active_stages = list(STAGES)

    if args.skip and args.skip in active_stages:
        active_stages.remove(args.skip)

    logger.info(f"Running stages: {' -> '.join(active_stages)}")

    t_start = time.perf_counter()
    classical_results: dict = {}
    quantum_results: dict = {}

    # If starting at evaluate/visualize without prior training in this run, load from disk
    needs_results = any(s in active_stages for s in ("evaluate", "visualize"))
    will_train_classical = "classical" in active_stages
    will_train_quantum = "quantum" in active_stages
    if needs_results and not (will_train_classical and will_train_quantum):
        logger.info("Loading saved model results from disk...")
        classical_results, quantum_results = load_eval_results(config)

    for stage in active_stages:
        t0 = time.perf_counter()
        if stage == "preprocess":
            stage_preprocess(config)
        elif stage == "label":
            stage_label(config)
        elif stage == "engineer":
            stage_engineer(config)
        elif stage == "classical":
            classical_results = stage_classical(config)
        elif stage == "quantum":
            quantum_results = stage_quantum(config)
        elif stage == "evaluate":
            stage_evaluate(config, classical_results, quantum_results)
        elif stage == "visualize":
            stage_visualize(config, classical_results, quantum_results)
        elif stage == "export_metrics":
            stage_export_metrics(config)
        elif stage == "visualize_extended":
            stage_visualize_extended(config)
        logger.info(f"Stage '{stage}' completed in {time.perf_counter() - t0:.1f}s")

    logger.info(f"Pipeline complete in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
