"""
Quantum ML models for climate condition classification.

Models:
  - QSVC: Quantum Support Vector Classifier using FidelityQuantumKernel
  - VQC: Variational Quantum Classifier using EfficientSU2 ansatz

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


class QuantumModelTrainer:
    """Unified training interface for QSVC and VQC."""

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

    def train_qsvc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_qubits: Optional[int] = None,
    ) -> dict:
        """Train Quantum SVC with FidelityQuantumKernel.

        Pipeline:
          1. Build ZZFeatureMap (4 qubits, reps=1, linear)
          2. StatevectorSampler -> ComputeUncompute -> FidelityQuantumKernel
          3. QSVC.fit(X_train, y_train)
          4. Evaluate on X_test

        Kernel matrix: O(n_train^2) circuits for training.
        With 400 samples and 4 qubits: ~seconds on CPU.
        """
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC

        n = n_qubits or self.n_qubits
        logger.info(f"Training QSVC ({n} qubits, {len(X_train)} training samples)...")
        t0 = time.perf_counter()

        feature_map = self._build_feature_map(n)
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        # Compute kernel matrix for visualization (subsample if large)
        n_vis = min(50, len(X_train))
        kernel_matrix = kernel.evaluate(X_train[:n_vis], X_train[:n_vis])

        preds = qsvc.predict(X_test)
        prediction_time = time.perf_counter() - t0 - training_time

        result = {
            "model": qsvc,
            "training_time": training_time,
            "prediction_time": prediction_time,
            "preds": preds,
            "kernel_matrix": kernel_matrix,
            "n_qubits": n,
            "n_train": len(X_train),
            "circuit_depth": feature_map.decompose().depth(),
        }
        logger.info(
            f"QSVC done: {training_time:.1f}s train, "
            f"circuit depth={result['circuit_depth']}"
        )
        self.results["qsvc"] = result
        return result

    def train_vqc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_qubits: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Train Variational Quantum Classifier.

        Pipeline:
          1. Build ZZFeatureMap (reps=1) + EfficientSU2 ansatz (reps=2, linear)
          2. VQC with COBYLA optimizer, pi-scaled initial point
          3. Track loss per iteration via callback
          4. Evaluate on X_test

        Expected training time: 10-30 min for 400 samples, COBYLA maxiter=150.
        """
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.algorithms import VQC

        n = n_qubits or self.n_qubits
        cfg_vqc = self.config["quantum_ml"]["vqc"]
        logger.info(f"Training VQC ({n} qubits, {len(X_train)} training samples)...")

        feature_map = self._build_feature_map(n)
        ansatz = self._build_ansatz(n)
        initial_point = self._initial_point(ansatz.num_parameters)

        loss_history = []

        def _callback(_nfev, _x, fx, _dx, _accept):
            loss_history.append(float(fx))
            if len(loss_history) % 25 == 0:
                logger.info(
                    f"VQC iter {len(loss_history)}: loss={fx:.4f}"
                )

        sampler = StatevectorSampler()
        maxiter = cfg_vqc["maxiter"]

        # SPSA: gradient-free optimizer designed for noisy objectives.
        # Replaces COBYLA which had a callback routing bug in qiskit-ml 0.9.0
        # (VQC.fit() never triggered the COBYLA callback, resulting in 0 iterations).
        if cfg_vqc["optimizer"] == "SPSA":
            from qiskit_algorithms.optimizers import SPSA
            optimizer = SPSA(maxiter=maxiter, callback=_callback)
        elif cfg_vqc["optimizer"] == "COBYLA":
            from qiskit_algorithms.optimizers import COBYLA
            optimizer = COBYLA(maxiter=maxiter)
        else:
            optimizer = None  # VQC defaults to SLSQP

        t0 = time.perf_counter()
        vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            callback=_callback,
        )
        vqc.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        preds = vqc.predict(X_test)
        prediction_time = time.perf_counter() - t0 - training_time

        result = {
            "model": vqc,
            "training_time": training_time,
            "prediction_time": prediction_time,
            "preds": preds,
            "loss_history": loss_history,
            "n_qubits": n,
            "n_params": ansatz.num_parameters,
            "n_train": len(X_train),
            "circuit_depth": (feature_map.compose(ansatz)).decompose().depth(),
        }
        final_loss = f"{float(loss_history[-1]):.4f}" if loss_history else "N/A"
        logger.info(
            f"VQC done: {training_time:.1f}s, {len(loss_history)} iters, "
            f"final loss={final_loss}"
        )
        self.results["vqc"] = result
        return result

    def qubit_scalability_analysis(
        self,
        X_train_full_pca: np.ndarray,
        y_train: np.ndarray,
        X_test_full_pca: np.ndarray,
        y_test: np.ndarray,
        qubit_range: Optional[list[int]] = None,
    ) -> "pd.DataFrame":
        """Train QSVC and VQC with varying qubit counts.

        For each n in qubit_range:
          - Uses first n PCA components of the pre-computed PCA data
          - Rescales to [0, pi]
          - Trains QSVC and VQC
          - Records accuracy, f1, training_time
        """
        import pandas as pd
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.preprocessing import MinMaxScaler

        if qubit_range is None:
            qubit_range = [2, 3, 4, 5, 6]

        records = []

        for n in qubit_range:
            logger.info(f"Qubit scalability: n_qubits={n}")

            # Subset PCA dims and rescale
            X_tr = X_train_full_pca[:, :n].copy()
            X_te = X_test_full_pca[:, :n].copy()
            q_scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_tr = q_scaler.fit_transform(X_tr)
            X_te = q_scaler.transform(X_te)

            for model_key in ["qsvc", "vqc"]:
                try:
                    if model_key == "qsvc":
                        r = self.train_qsvc(X_tr, y_train, X_te, y_test,
                                             n_qubits=n)
                    else:
                        r = self.train_vqc(X_tr, y_train, X_te, y_test,
                                            n_qubits=n)

                    preds = r["preds"]
                    records.append({
                        "model": model_key.upper(),
                        "n_qubits": n,
                        "accuracy": float(accuracy_score(y_test, preds)),
                        "f1_macro": float(f1_score(y_test, preds, average="macro",
                                                    zero_division=0)),
                        "training_time": r["training_time"],
                    })
                except Exception as e:
                    logger.warning(f"Failed {model_key} n={n}: {e}")
                    records.append({
                        "model": model_key.upper(),
                        "n_qubits": n,
                        "accuracy": np.nan,
                        "f1_macro": np.nan,
                        "training_time": np.nan,
                    })

        return pd.DataFrame(records)

    def run_on_ibm_hardware(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test_subset: np.ndarray,
        y_test_subset: np.ndarray,
        backend_name: Optional[str] = None,
    ) -> dict:
        """Run QSVC inference on IBM Quantum hardware.

        Due to 10 min/month budget:
          - Training kernel matrix computed on simulator (StatevectorSampler)
          - Hardware is used only to evaluate the test kernel rows (20 samples max)
          - Circuits are transpiled to ISA before submission (required post-March 2024)

        Strategy:
          1. Build feature map + ComputeUncompute fidelity on StatevectorSampler
          2. Fit QSVC on full training set (sim kernel — fast)
          3. For test prediction, build a hardware-backed kernel, evaluate only
             the test-vs-train kernel matrix rows needed for predict(), then
             override the decision function manually.

        Returns hardware predictions and noise comparison vs simulator.
        """
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit.primitives import StatevectorSampler as LocalSampler
        import os

        cfg_ibm_local = self.config["quantum_ml"]["ibm_quantum"]
        if cfg_ibm_local["channel"] == "ibm_cloud":
            token = os.environ.get("IBM_CLOUD_API_KEY")
            if not token:
                raise ValueError("Set IBM_CLOUD_API_KEY environment variable")
        else:
            token = os.environ.get("IBM_QUANTUM_TOKEN")
            if not token:
                raise ValueError("Set IBM_QUANTUM_TOKEN environment variable")

        cfg_ibm = self.config["quantum_ml"]["ibm_quantum"]
        max_samples = cfg_ibm["max_test_samples"]
        X_hw = X_test_subset[:max_samples]
        y_hw = y_test_subset[:max_samples]

        logger.info(f"Connecting to IBM Quantum (channel={cfg_ibm['channel']})...")
        service = QiskitRuntimeService(
            channel=cfg_ibm["channel"],
            token=token,
            instance=cfg_ibm["instance"],
        )

        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)

        logger.info(f"Using backend: {backend.name}")

        t0 = time.perf_counter()

        # Step 1: Fit QSVC on simulator (fast, no hardware quota used)
        feature_map = self._build_feature_map()
        sim_fidelity = ComputeUncompute(sampler=LocalSampler())
        sim_kernel = FidelityQuantumKernel(fidelity=sim_fidelity, feature_map=feature_map)
        qsvc_hw = QSVC(quantum_kernel=sim_kernel)
        logger.info(f"Fitting QSVC on simulator ({len(X_train)} samples)...")
        qsvc_hw.fit(X_train, y_train)
        sim_fit_time = time.perf_counter() - t0
        logger.info(f"Simulator fit done in {sim_fit_time:.1f}s")

        # Step 2: Build hardware-backed kernel with transpiled circuits
        logger.info("Building transpiled hardware kernel...")
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        hw_sampler = SamplerV2(mode=backend)
        hw_fidelity = ComputeUncompute(sampler=hw_sampler, pass_manager=pm)
        hw_kernel = FidelityQuantumKernel(fidelity=hw_fidelity, feature_map=feature_map)

        # Step 3: Evaluate K(X_hw, support_vectors) row-by-row on hardware.
        # We only need kernel values against support vectors (not all 400 train),
        # and submit one test sample per job to stay under the 10M shot limit.
        support_indices = qsvc_hw.support_
        X_sv = X_train[support_indices]
        n_sv = len(X_sv)
        logger.info(
            f"Evaluating hardware kernel: {len(X_hw)} test samples x "
            f"{n_sv} support vectors (one job per test sample)..."
        )

        K_test_sv_hw = np.zeros((len(X_hw), n_sv))
        for i, x in enumerate(X_hw):
            row = hw_kernel.evaluate(x.reshape(1, -1), X_sv)
            K_test_sv_hw[i] = row[0]
            logger.info(f"  HW kernel row {i+1}/{len(X_hw)} done")

        # Step 4: Decision function using hardware kernel values
        # dual_coef_ * K(x, sv) + intercept_ => class scores
        decision = K_test_sv_hw @ qsvc_hw.dual_coef_.T + qsvc_hw.intercept_
        if decision.shape[1] == 1:
            hw_preds = (decision.ravel() > 0).astype(int)
        else:
            hw_preds = np.argmax(decision, axis=1)

        elapsed = time.perf_counter() - t0

        # Simulator predictions for comparison
        sim_preds = None
        if "qsvc" in self.results and self.results["qsvc"].get("model") is not None:
            try:
                sim_preds = self.results["qsvc"]["model"].predict(X_hw)
            except Exception:
                pass

        result = {
            "backend": backend.name,
            "n_test": len(X_hw),
            "hw_preds": hw_preds,
            "sim_preds": sim_preds,
            "y_true": y_hw,
            "elapsed_s": elapsed,
        }
        if sim_preds is not None:
            agreement = float(np.mean(hw_preds == sim_preds))
            result["hw_sim_agreement"] = agreement
            logger.info(f"Hardware vs Simulator agreement: {agreement:.1%}")
        return result

    def draw_circuits(self, output_dir: Path) -> None:
        """Draw and save feature map and VQC circuit diagrams."""
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
            (full_circuit, "vqc_full_circuit"),
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

    trainer = QuantumModelTrainer(cfg)
    trainer.draw_circuits(Path(cfg["paths"]["figures"]))

    qsvc_result = trainer.train_qsvc(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
    )
    print(f"QSVC done: {qsvc_result['training_time']:.1f}s")

    vqc_result = trainer.train_vqc(
        quantum["X_train_q"], quantum["y_train_q"],
        quantum["X_test_q"], quantum["y_test_q"],
    )
    print(f"VQC done: {vqc_result['training_time']:.1f}s")
    trainer.save_models(Path(cfg["paths"]["models"]))
