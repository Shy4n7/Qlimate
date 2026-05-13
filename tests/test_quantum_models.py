"""
Unit tests for src/models/quantum.py — QuantumModelTrainer.

All tests use synthetic data only. No IBM credentials required.
Uses StatevectorSampler (local simulator) for all quantum computations.

Synthetic dataset:
  - 20 training samples, 10 test samples
  - 4 features, values in [0, π]
  - 2 classes (10 per class in train, 5 per class in test)
"""

import numpy as np
import pytest

from src.models.quantum import QuantumModelTrainer

# ---------------------------------------------------------------------------
# Minimal config — no dependency on config/config.yaml
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "quantum_ml": {
        "n_qubits": 4,
        "qsvc": {"feature_map_reps": 1, "entanglement": "linear"},
        "vqc": {
            "feature_map_reps": 1,
            "ansatz_reps": 1,
            "entanglement": "linear",
            "optimizer": "SPSA",
            "maxiter": 5,
            "initial_point_scale": 0.5,
        },
        "qsvc_pauli": {
            "feature_map_reps": 1,
            "entanglement": "linear",
            "paulis": ["Z", "ZZ"],
        },
        "vqc_real_amplitudes": {
            "ansatz_reps": 1,
            "entanglement": "linear",
            "optimizer": "SPSA",
            "maxiter": 5,
        },
    }
}

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """
    20 training samples, 10 test samples, 4 features in [0, π], 2 classes.
    10 samples per class in train, 5 per class in test.
    """
    rng = np.random.default_rng(42)

    X_train = rng.random((20, 4)) * np.pi
    y_train = np.array([0] * 10 + [1] * 10)

    X_test = rng.random((10, 4)) * np.pi
    y_test = np.array([0] * 5 + [1] * 5)

    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def trainer():
    """QuantumModelTrainer initialised with the minimal inline config."""
    return QuantumModelTrainer(config=MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# QSVC tests (fast — kernel-based, no variational optimisation)
# ---------------------------------------------------------------------------


def test_train_qsvc_returns_required_keys(trainer, synthetic_data):
    """train_qsvc result must contain model, training_time, preds, kernel_matrix, n_qubits."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_qsvc(X_train, y_train, X_test, y_test)

    required_keys = {"model", "training_time", "preds", "kernel_matrix", "n_qubits"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_kernel_matrix_shape_and_range(trainer, synthetic_data):
    """kernel_matrix must be square and all values must be in [0.0, 1.0].

    A small floating-point tolerance (1e-10) is applied to account for
    numerical precision in the quantum fidelity computation — values like
    -1.34e-15 are effectively zero and should not fail the test.
    """
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_qsvc(X_train, y_train, X_test, y_test)

    km = result["kernel_matrix"]
    assert km.shape[0] == km.shape[1], (
        f"kernel_matrix is not square: shape={km.shape}"
    )
    tol = 1e-10
    assert float(km.min()) >= -tol, (
        f"kernel_matrix has values significantly below 0: min={km.min()}"
    )
    assert float(km.max()) <= 1.0 + tol, (
        f"kernel_matrix has values significantly above 1: max={km.max()}"
    )


def test_train_qsvc_pauli_returns_required_keys(trainer, synthetic_data):
    """train_qsvc_pauli result must contain model, training_time, preds, kernel_matrix, n_qubits, feature_map_type."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_qsvc_pauli(X_train, y_train, X_test, y_test)

    required_keys = {
        "model", "training_time", "preds", "kernel_matrix", "n_qubits", "feature_map_type"
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_train_qsvc_pauli_feature_map_type(trainer, synthetic_data):
    """train_qsvc_pauli must record feature_map_type == 'PauliFeatureMap'."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_qsvc_pauli(X_train, y_train, X_test, y_test)

    assert result["feature_map_type"] == "PauliFeatureMap"


# ---------------------------------------------------------------------------
# Circuit builder tests (no training — just circuit construction)
# ---------------------------------------------------------------------------


def test_build_feature_map_qubits(trainer):
    """_build_feature_map(n_qubits=4) must return a circuit with num_qubits == 4."""
    circuit = trainer._build_feature_map(n_qubits=4)
    assert circuit.num_qubits == 4


def test_build_ansatz_qubits_and_params(trainer):
    """_build_ansatz(n_qubits=4) must return a circuit with num_qubits==4 and num_parameters>0."""
    circuit = trainer._build_ansatz(n_qubits=4)
    assert circuit.num_qubits == 4
    assert circuit.num_parameters > 0


# ---------------------------------------------------------------------------
# VQC tests — marked slow (variational optimisation takes minutes on real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_vqc_returns_required_keys(trainer, synthetic_data):
    """train_vqc result must contain model, training_time, preds, n_params."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_vqc(X_train, y_train, X_test, y_test)

    required_keys = {"model", "training_time", "preds", "n_params"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


@pytest.mark.slow
def test_train_vqc_real_amplitudes_returns_required_keys(trainer, synthetic_data):
    """train_vqc_real_amplitudes result must contain model, training_time, preds, n_params, ansatz_type."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_vqc_real_amplitudes(X_train, y_train, X_test, y_test)

    required_keys = {"model", "training_time", "preds", "n_params", "ansatz_type"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


@pytest.mark.slow
def test_train_vqc_real_amplitudes_ansatz_type(trainer, synthetic_data):
    """train_vqc_real_amplitudes must record ansatz_type == 'RealAmplitudes'."""
    X_train, y_train, X_test, y_test = synthetic_data
    result = trainer.train_vqc_real_amplitudes(X_train, y_train, X_test, y_test)

    assert result["ansatz_type"] == "RealAmplitudes"
