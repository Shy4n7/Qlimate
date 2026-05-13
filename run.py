"""
Qlimate — regression pipeline runner.

Predicts T2M surface temperature (°C) for Indian states using classical
and quantum ML regressors. Quantum models are precomputed offline for all
(state × month × year) combinations and served from a JSON lookup table.

Stages:
  1. preprocess         — aggregate MERRA-2 grid to India states
  2. engineer           — feature engineering + regression train/val/test splits
  3. classical          — train XGBoost, RF, Ridge, GradientBoosting, NN regressors
  4. quantum            — train QSVR and VQR
  5. precompute         — precompute all quantum predictions offline (runs once, ~30–60 min)
  6. evaluate           — unified regression comparison report (MAE, RMSE, R²)
  7. visualize          — static figures + interactive dashboard
  8. export_metrics     — export metrics to results/metrics/
  9. visualize_extended — extended figures (loss curves, kernel matrix, etc.)

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

STAGES = [
    "preprocess",
    "engineer",
    "classical",
    "quantum",
    "precompute",
    "evaluate",
    "visualize",
    "export_metrics",
    "visualize_extended",
]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_preprocess(config: dict) -> None:
    from src.data.preprocess import process_all
    logger.info("=== STAGE: preprocess ===")
    df = process_all(config)
    logger.info(f"Preprocessed dataset: {df.shape[0]} rows, {df.shape[1]} columns")


def stage_engineer(config: dict) -> None:
    import pandas as pd
    from src.features.engineering import (
        prepare_regression_splits,
        prepare_quantum_regression_subset,
        save_artifacts,
    )
    logger.info("=== STAGE: engineer ===")

    processed_dir = Path(config["paths"]["processed_data"])
    states_path = processed_dir / "merra2_india_states.csv"
    if not states_path.exists():
        raise FileNotFoundError(f"Run 'preprocess' stage first: {states_path}")

    df = pd.read_csv(states_path)
    splits = prepare_regression_splits(df, config)
    quantum = prepare_quantum_regression_subset(
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
    logger.info("Saved regression_splits.npz and quantum_regression_splits.npz")


def stage_classical(config: dict) -> dict:
    import torch
    from src.features.engineering import load_splits
    from src.models.classical import ClassicalRegressorTrainer
    from src.evaluation.metrics import evaluate_regressor
    logger.info("=== STAGE: classical ===")

    splits, _ = load_splits(config)
    trainer = ClassicalRegressorTrainer(config)

    results = trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )

    models_dir = Path(config["paths"]["models"])
    trainer.save_models(models_dir)

    def make_predict_fn(name: str, res: dict, trainer: "ClassicalRegressorTrainer"):
        if name == "neural_network_regressor":
            model = res["model"]
            device = trainer.device

            def nn_predict(X: np.ndarray) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    t = torch.tensor(X, dtype=torch.float32).to(device)
                    return model(t).squeeze(1).cpu().numpy()

            return nn_predict
        return lambda X, m=res["model"]: m.predict(X)

    eval_results = {}
    for name, res in results.items():
        if res is None:
            logger.warning(f"  {name}: skipped (returned None)")
            continue
        metrics = evaluate_regressor(
            make_predict_fn(name, res, trainer),
            splits["X_test"], splits["y_test"], name,
        )
        metrics["training_time"] = res.get("training_time", 0)
        metrics["n_train"] = len(splits["X_train"])
        eval_results[name] = metrics
        logger.info(
            f"  {name:35s}  mae={metrics['mae']:.4f}°C  "
            f"rmse={metrics['rmse']:.4f}°C  time={res['training_time']:.1f}s"
        )

    return eval_results


def stage_quantum(config: dict) -> dict:
    from src.features.engineering import load_splits
    from src.models.quantum import QuantumRegressorTrainer
    from src.evaluation.metrics import evaluate_regressor
    logger.info("=== STAGE: quantum ===")

    splits, quantum = load_splits(config)
    trainer = QuantumRegressorTrainer(config)

    figures_dir = Path(config["paths"]["figures"])
    trainer.draw_circuits(figures_dir)

    models_dir = Path(config["paths"]["models"])

    qsvr_result = trainer.train_qsvr(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
        models_dir=models_dir,
    )
    trainer.save_models(models_dir)

    vqr_result = trainer.train_vqr(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
        models_dir=models_dir,
    )
    trainer.save_models(models_dir)

    eval_results = {}
    for name, res in [
        ("QSVR", qsvr_result),
        ("VQR", vqr_result),
    ]:
        metrics = evaluate_regressor(
            None,
            quantum["X_test_q"], quantum["y_test_q"], name,
            precomputed_preds=res["preds"],
        )
        metrics["training_time"] = res["training_time"]
        metrics["n_train"] = res.get("n_train", len(quantum["X_train_q"]))
        eval_results[name] = metrics
        logger.info(
            f"  {name:25s}  mae={metrics['mae']:.4f}°C  "
            f"rmse={metrics['rmse']:.4f}°C  time={res['training_time']:.1f}s"
        )

    return eval_results


def stage_precompute(config: dict) -> None:
    """Run offline quantum prediction precomputation for all (state × month × year) combos.

    This stage runs once after quantum model training and may take 30–60 minutes on CPU.
    It generates results/quantum_predictions.json covering all 13,776 combinations
    (28 states × 12 months × 41 years: 1995–2035).

    Quantum models cannot run inference in real time; this precomputed lookup table
    is loaded by the FastAPI server at startup and served in milliseconds.
    """
    import subprocess
    logger.info("=== STAGE: precompute ===")
    logger.info(
        "Running offline quantum prediction precomputation. "
        "This runs once and may take 30–60 minutes on CPU."
    )
    logger.info(
        "Generating predictions for all 28 states × 12 months × 41 years "
        "(1995–2035) = 13,776 combinations."
    )

    script_path = Path("scripts") / "precompute_quantum.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Precompute script not found: {script_path}. "
            "Run task 9 (create scripts/precompute_quantum.py) first."
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("precompute_quantum.py failed with non-zero exit code")
        raise RuntimeError("Quantum precomputation failed — see output above for details.")
    else:
        logger.info("Quantum precomputation complete. Results saved to results/quantum_predictions.json")


def load_eval_results(config: dict) -> tuple[dict, dict]:
    """Re-evaluate saved classical and quantum regression models on test data."""
    import joblib
    import torch
    from src.features.engineering import load_splits
    from src.models.classical import ClimateRegressionNN
    from src.evaluation.metrics import evaluate_regressor

    splits, quantum = load_splits(config)
    models_dir = Path(config["paths"]["models"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_classical = len(splits["X_train"])

    classical_results: dict = {}

    # sklearn-serialized regression models (joblib .pkl)
    sklearn_models = [
        ("xgboost_regressor", "xgboost_regressor.pkl"),
        ("random_forest_regressor", "random_forest_regressor.pkl"),
        ("ridge_regression", "ridge_regression.pkl"),
        ("gradient_boosting_regressor", "gradient_boosting_regressor.pkl"),
    ]
    for name, fname in sklearn_models:
        path = models_dir / fname
        if not path.exists():
            logger.debug(f"Skipping {name} — {path} not found")
            continue
        model = joblib.load(path)
        metrics = evaluate_regressor(
            lambda X, m=model: m.predict(X),
            splits["X_test"], splits["y_test"], name,
        )
        metrics["training_time"] = 0  # not captured at load time
        metrics["n_train"] = n_train_classical
        classical_results[name] = metrics

    # Neural network regressor (PyTorch state dict)
    nn_path = models_dir / "neural_network_regressor_state_dict.pt"
    if nn_path.exists():
        meta = joblib.load(models_dir / "neural_network_regressor_meta.pkl")
        n_features = splits["X_test"].shape[1]
        cfg_nn = config["classical_ml"]["neural_net"]
        model = ClimateRegressionNN(
            n_features, cfg_nn["hidden_dims"], cfg_nn["dropout"]
        ).to(device)
        model.load_state_dict(torch.load(nn_path, map_location=device))
        model.eval()

        def nn_predict(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32).to(device)
                return model(t).squeeze(1).cpu().numpy()

        metrics = evaluate_regressor(
            nn_predict, splits["X_test"], splits["y_test"],
            "neural_network_regressor",
        )
        metrics["training_time"] = 0
        metrics["n_train"] = n_train_classical
        classical_results["neural_network_regressor"] = metrics

    # Quantum regression models — load from saved meta (predictions pre-computed)
    quantum_results: dict = {}
    quantum_model_names = [
        ("qsvr", "QSVR"),
        ("vqr", "VQR"),
    ]
    for fname_base, display_name in quantum_model_names:
        meta_path = models_dir / f"{fname_base}_meta.pkl"
        if not meta_path.exists():
            logger.debug(f"Skipping {display_name} — {meta_path} not found")
            continue
        meta = joblib.load(meta_path)
        preds = meta.get("preds")
        if preds is not None:
            metrics = evaluate_regressor(
                None, quantum["X_test_q"], quantum["y_test_q"],
                display_name, precomputed_preds=preds,
            )
            metrics["training_time"] = meta.get("training_time", 0)
            metrics["n_train"] = meta.get("n_train", len(quantum["X_train_q"]))
            metrics["loss_history"] = meta.get("loss_history", [])
            quantum_results[display_name] = metrics

    return classical_results, quantum_results


def stage_evaluate(config: dict, classical_results: dict, quantum_results: dict) -> None:
    from src.evaluation.metrics import compare_regressors
    logger.info("=== STAGE: evaluate ===")

    all_results = {**classical_results, **quantum_results}
    comparison = compare_regressors(all_results)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    logger.info("\nModel comparison (sorted by RMSE ascending):")
    logger.info(comparison.to_string(index=False))

    # Log best model summary
    if not comparison.empty:
        best = comparison.iloc[0]
        logger.info(
            f"\nBest model: {best['model']}  "
            f"rmse={best['rmse']:.4f}°C  mae={best['mae']:.4f}°C  r2={best['r2']:.4f}"
        )


def stage_visualize(config: dict, classical_results: dict, quantum_results: dict) -> None:
    from src.features.engineering import load_splits
    from src.evaluation.metrics import compare_regressors
    logger.info("=== STAGE: visualize ===")

    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = {**classical_results, **quantum_results}

    try:
        comparison = compare_regressors(all_results)
        from src.visualization.static_plots import (
            plot_model_comparison_bar,
            create_summary_figure,
        )
        plot_model_comparison_bar(comparison, figures_dir / "model_comparison.png")
        create_summary_figure(comparison, all_results, [], figures_dir / "summary.png")
    except Exception as e:
        logger.warning(f"Static plots failed (non-fatal): {e}")

    try:
        from src.visualization.interactive import save_html_dashboard
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
        logger.info("Interactive dashboard saved to results/dashboard.html")
    except Exception as e:
        logger.warning(f"Interactive dashboard failed (non-fatal): {e}")

    logger.info(f"Figures saved to {figures_dir}")


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
    logger.info("=== STAGE: visualize_extended ===")

    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.visualization.optimization_plots import (
            plot_nn_loss_curves, plot_vqc_no_convergence, plot_combined_optimization,
        )
        _, quantum = load_splits(config)
        nn_meta = joblib.load(Path(config["paths"]["models"]) / "neural_network_regressor_meta.pkl")
        vqr_meta = joblib.load(Path(config["paths"]["models"]) / "vqr_meta.pkl")
        vqr_meta["maxiter_configured"] = config["quantum_ml"]["vqc"]["maxiter"]

        plot_nn_loss_curves(nn_meta["train_losses"], nn_meta["val_losses"],
                            figures_dir / "nn_loss_curves.png")
        plot_vqc_no_convergence(vqr_meta, figures_dir / "vqr_no_convergence.png")
        plot_combined_optimization(nn_meta, vqr_meta, figures_dir / "optimization_comparison.png")
    except Exception as e:
        logger.warning(f"Optimization plots failed (non-fatal): {e}")

    try:
        from src.visualization.scaling_plots import (
            plot_quantum_scaling, plot_classical_vs_quantum_scaling,
        )
        plot_quantum_scaling(figures_dir / "quantum_scaling.png")
        plot_classical_vs_quantum_scaling(figures_dir / "scaling_comparison.png")
    except Exception as e:
        logger.warning(f"Scaling plots failed (non-fatal): {e}")

    try:
        pca = joblib.load(Path(config["paths"]["models"]) / "pca_model.pkl")
        from src.visualization.feature_flow import plot_feature_compression_diagram
        plot_feature_compression_diagram(pca.explained_variance_ratio_.tolist(),
                                         figures_dir / "feature_compression.png")
    except Exception as e:
        logger.warning(f"Feature compression diagram failed (non-fatal): {e}")

    try:
        from src.visualization.pipeline_breakdown import plot_pipeline_timing
        plot_pipeline_timing(figures_dir / "pipeline_breakdown.png")
    except Exception as e:
        logger.warning(f"Pipeline timing plot failed (non-fatal): {e}")

    try:
        _, quantum = load_splits(config)
        from src.visualization.kernel_visualization import (
            kernel_matrix_from_model, plot_kernel_matrix_heatmap,
        )
        logger.info("Computing kernel matrix (may take ~30s)...")
        K = kernel_matrix_from_model(quantum["X_train_q"], n_vis=50, config=config)
        plot_kernel_matrix_heatmap(K, figures_dir / "kernel_matrix.png")
    except Exception as e:
        logger.warning(f"Kernel matrix visualization failed (non-fatal): {e}")

    logger.info(f"Extended figures saved to {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qlimate regression pipeline runner")
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
        elif stage == "engineer":
            stage_engineer(config)
        elif stage == "classical":
            classical_results = stage_classical(config)
        elif stage == "quantum":
            quantum_results = stage_quantum(config)
        elif stage == "precompute":
            stage_precompute(config)
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
