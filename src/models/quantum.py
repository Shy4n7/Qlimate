"""
Quantum ML models for climate temperature regression.

Models:
  - QSVR: Quantum Support Vector Regressor using FidelityQuantumKernel
  - VQR:  Variational Quantum Regressor using EfficientSU2 ansatz

Uses Qiskit 2.x APIs:
  - ZZFeatureMap class (qiskit.circuit.library)
  - EfficientSU2 class (qiskit.circuit.library)
  - StatevectorSampler (V2 primitive)
  - FidelityQuantumKernel + ComputeUncompute

Barren plateau mitigation:
  - 4 qubits max
  - Linear entanglement in ansatz
  - Shallow circuits (feature_map reps=1, ansatz reps=2)
  - Non-random pi-scaled parameter initialization
"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable

import joblib
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class QuantumRegressorTrainer:
    """Unified training interface for QSVR and VQR."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: str = "config/config.yaml"):
        self.config = config or load_config(config_path)
        self.results: dict = {}
        self.n_qubits = self.config["quantum_ml"]["n_qubits"]

    def _build_feature_map(self, n_qubits: Optional[int] = None):
        """Build ZZFeatureMap for angle encoding."""
        from qiskit.circuit.library import ZZFeatureMap
        n = n_qubits or self.n_qubits
        reps = self.config["quantum_ml"]["qsvc"]["feature_map_reps"]
        ent = self.config["quantum_ml"]["qsvc"]["entanglement"]
        return ZZFeatureMap(feature_dimension=n, reps=reps, entanglement=ent)

    def _build_ansatz(self, n_qubits: Optional[int] = None):
        """Build EfficientSU2 ansatz with linear entanglement."""
        from qiskit.circuit.library import EfficientSU2
        n = n_qubits or self.n_qubits
        reps = self.config["quantum_ml"]["vqc"]["ansatz_reps"]
        ent = self.config["quantum_ml"]["vqc"]["entanglement"]
        return EfficientSU2(num_qubits=n, reps=reps, entanglement=ent)

    def _initial_point(self, n_params: int) -> np.ndarray:
        """Non-random pi-scaled parameter initialization to avoid barren plateaus."""
        scale = self.config["quantum_ml"]["vqc"]["initial_point_scale"]
        rng = np.random.default_rng(42)
        return scale * np.pi * rng.random(n_params)

    def train_qsvr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_qubits: Optional[int] = None,
        models_dir: Optional[Path] = None,
    ) -> dict:
        """Train Quantum SVR with FidelityQuantumKernel.

        Pipeline:
          1. Build ZZFeatureMap (4 qubits, reps=1, linear)
          2. StatevectorSampler -> ComputeUncompute -> FidelityQuantumKernel
          3. QSVR.fit(X_train, y_train)
          4. Predict on X_test (continuous float temperatures in °C)

        Kernel matrix: O(n_train^2) circuits for training.
        With 400 samples and 4 qubits: ~seconds on CPU.

        Returns:
            dict with keys: model, training_time, preds, kernel_matrix,
                            n_qubits, n_train
        """
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVR

        n = n_qubits or self.n_qubits
        logger.info(f"Training QSVR ({n} qubits, {len(X_train)} training samples)...")
        t0 = time.perf_counter()

        feature_map = self._build_feature_map(n)
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        qsvr = QSVR(quantum_kernel=kernel)
        qsvr.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        # Compute kernel matrix for visualization (subsample if large)
        n_vis = min(50, len(X_train))
        kernel_matrix = kernel.evaluate(X_train[:n_vis], X_train[:n_vis])

        preds = qsvr.predict(X_test)

        result = {
            "model": qsvr,
            "training_time": training_time,
            "preds": preds,
            "kernel_matrix": kernel_matrix,
            "n_qubits": n,
            "n_train": len(X_train),
        }
        logger.info(f"QSVR done: {training_time:.1f}s train")
        self.results["qsvr"] = result

        # Auto-save metadata after training completes (Requirement 4.6)
        if models_dir is not None:
            self._auto_save_meta("qsvr", result, Path(models_dir))

        return result

    def train_vqr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_qubits: Optional[int] = None,
        models_dir: Optional[Path] = None,
    ) -> dict:
        """Train Variational Quantum Regressor.

        Pipeline:
          1. Build ZZFeatureMap (reps=1) + EfficientSU2 ansatz (reps=2, linear)
          2. VQR with SPSA optimizer (maxiter=150), pi-scaled initial point
          3. Track loss per iteration via SPSA callback
          4. Predict on X_test (continuous float temperatures in °C)

        Expected training time: 20-60 min for 400 samples, SPSA maxiter=150.

        Returns:
            dict with keys: model, training_time, preds, loss_history,
                            n_qubits, n_params, n_train
        """
        from qiskit_machine_learning.algorithms import VQR

        n = n_qubits or self.n_qubits
        cfg_vqc = self.config["quantum_ml"]["vqc"]
        logger.info(f"Training VQR ({n} qubits, {len(X_train)} training samples)...")

        feature_map = self._build_feature_map(n)
        ansatz = self._build_ansatz(n)
        initial_point = self._initial_point(ansatz.num_parameters)

        loss_history = []

        def _callback(_nfev, _x, fx, _dx, _accept):
            loss_history.append(float(fx))
            if len(loss_history) % 25 == 0:
                logger.info(
                    f"VQR iter {len(loss_history)}: loss={fx:.4f}"
                )

        # VQR uses an Estimator primitive (not Sampler); defaults to reference Estimator when None.
        maxiter = cfg_vqc["maxiter"]

        from qiskit_algorithms.optimizers import SPSA
        optimizer = SPSA(maxiter=maxiter, callback=_callback)

        t0 = time.perf_counter()
        # Note: _callback (5-arg SPSA signature) is passed to SPSA, not VQR.
        # VQR's own callback expects (weights, value) — a different 2-arg signature.
        # Loss history is captured via the SPSA-level callback above.
        vqr = VQR(
            num_qubits=n,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )
        vqr.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        preds = vqr.predict(X_test)

        result = {
            "model": vqr,
            "training_time": training_time,
            "preds": preds,
            "loss_history": loss_history,
            "n_qubits": n,
            "n_params": ansatz.num_parameters,
            "n_train": len(X_train),
        }
        final_loss = f"{float(loss_history[-1]):.4f}" if loss_history else "N/A"
        logger.info(
            f"VQR done: {training_time:.1f}s, {len(loss_history)} iters, "
            f"final loss={final_loss}"
        )
        self.results["vqr"] = result

        # Auto-save metadata after training completes (Requirement 4.6)
        if models_dir is not None:
            self._auto_save_meta("vqr", result, Path(models_dir))

        return result

    def _auto_save_meta(self, name: str, result: dict, models_dir: Path) -> None:
        """Auto-save training metadata to <name>_meta.pkl after training."""
        models_dir.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in result.items() if k not in ("model", "kernel_matrix")}
        meta_path = models_dir / f"{name}_meta.pkl"
        try:
            joblib.dump(meta, meta_path)
            logger.info(f"Auto-saved {name} metadata to {meta_path}")
        except Exception as e:
            logger.warning(f"Could not auto-save {name} metadata: {e}")

    def draw_circuits(self, output_dir: Path) -> None:
        """Draw and save feature map and VQR circuit diagrams."""
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")

        output_dir.mkdir(parents=True, exist_ok=True)

        feature_map = self._build_feature_map()
        ansatz = self._build_ansatz()
        full_circuit = feature_map.compose(ansatz)

        for circuit, name in [
            (feature_map, "feature_map"),
            (ansatz, "ansatz"),
            (full_circuit, "vqr_full_circuit"),
        ]:
            try:
                fig = circuit.draw("mpl", fold=-1, style="clifford")
                fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved circuit diagram: {name}.png")
            except Exception as e:
                logger.warning(f"Could not draw circuit '{name}': {e}")

    def save_models(self, models_dir: Path) -> None:
        """Save quantum models and metadata."""
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, result in self.results.items():
            meta = {k: v for k, v in result.items() if k not in ("model", "kernel_matrix")}
            joblib.dump(meta, models_dir / f"{name}_meta.pkl")
            try:
                joblib.dump(result["model"], models_dir / f"{name}.pkl")
            except Exception as e:
                logger.warning(f"Could not save {name} model object: {e}")
        logger.info(f"Saved quantum model metadata to {models_dir}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    from src.features.engineering import load_splits
    splits, quantum = load_splits(cfg)

    models_dir = Path(cfg["paths"]["models"])
    trainer = QuantumRegressorTrainer(cfg)
    trainer.draw_circuits(Path(cfg["paths"]["figures"]))

    qsvr_result = trainer.train_qsvr(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
        models_dir=models_dir,
    )
    print(f"QSVR done: {qsvr_result['training_time']:.1f}s")

    vqr_result = trainer.train_vqr(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
        models_dir=models_dir,
    )
    print(f"VQR done: {vqr_result['training_time']:.1f}s")
    trainer.save_models(models_dir)
