"""
Classical ML models for climate condition classification.

Models:
  - Random Forest (sklearn)
  - SVM with RBF kernel (sklearn)
  - XGBoost
  - Neural Network (PyTorch, GPU-accelerated)

All models expose a common interface: train_<model>() returns a result dict
with model, metrics, training_time, val_metrics.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PyTorch Neural Network
# ---------------------------------------------------------------------------

class ClimateNN(nn.Module):
    """Feedforward MLP for 5-class climate condition classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], n_classes: int,
                 dropout: list[float]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim, drop in zip(hidden_dims, dropout):
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(drop),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Classical Model Trainer
# ---------------------------------------------------------------------------

class ClassicalModelTrainer:
    """Unified training interface for all classical models."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: str = "config/config.yaml"):
        self.config = config or load_config(config_path)
        self.results: dict = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    def train_random_forest(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        cfg = self.config["classical_ml"]["random_forest"]
        param_grid = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_split": cfg["min_samples_split"],
        }
        logger.info("Training Random Forest with GridSearchCV...")
        t0 = time.perf_counter()

        base = RandomForestClassifier(
            random_state=self.config["data_split"]["random_state"],
            n_jobs=-1,
            class_weight="balanced",
        )
        gs = GridSearchCV(base, param_grid, cv=cfg["cv_folds"],
                          scoring="f1_macro", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        training_time = time.perf_counter() - t0
        model = gs.best_estimator_

        val_preds = model.predict(X_val)
        result = {
            "model": model,
            "best_params": gs.best_params_,
            "training_time": training_time,
            "val_preds": val_preds,
            "feature_importances": model.feature_importances_,
        }
        logger.info(f"RF done in {training_time:.1f}s. Best params: {gs.best_params_}")
        self.results["random_forest"] = result
        return result

    def train_svm(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        cfg = self.config["classical_ml"]["svm"]
        param_grid = {
            "C": cfg["C"],
            "gamma": cfg["gamma"],
        }
        logger.info("Training SVM with GridSearchCV...")
        t0 = time.perf_counter()

        base = SVC(kernel="rbf", class_weight="balanced",
                   random_state=self.config["data_split"]["random_state"],
                   probability=True)
        gs = GridSearchCV(base, param_grid, cv=cfg["cv_folds"],
                          scoring="f1_macro", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        training_time = time.perf_counter() - t0
        model = gs.best_estimator_

        val_preds = model.predict(X_val)
        result = {
            "model": model,
            "best_params": gs.best_params_,
            "training_time": training_time,
            "val_preds": val_preds,
        }
        logger.info(f"SVM done in {training_time:.1f}s. Best params: {gs.best_params_}")
        self.results["svm"] = result
        return result

    def train_xgboost(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        cfg = self.config["classical_ml"]["xgboost"]
        param_grid = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
        }
        logger.info("Training XGBoost with GridSearchCV...")
        t0 = time.perf_counter()

        tree_method = "hist"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        base = XGBClassifier(
            tree_method=tree_method,
            device=device_str,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.config["data_split"]["random_state"],
            verbosity=0,
        )
        gs = GridSearchCV(base, param_grid, cv=cfg["cv_folds"],
                          scoring="f1_macro", n_jobs=1, verbose=0)
        gs.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               verbose=False)

        training_time = time.perf_counter() - t0
        model = gs.best_estimator_

        val_preds = model.predict(X_val)
        result = {
            "model": model,
            "best_params": gs.best_params_,
            "training_time": training_time,
            "val_preds": val_preds,
            "feature_importances": model.feature_importances_,
        }
        logger.info(f"XGBoost done in {training_time:.1f}s. Best params: {gs.best_params_}")
        self.results["xgboost"] = result
        return result

    def train_neural_network(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        cfg = self.config["classical_ml"]["neural_net"]
        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]

        logger.info(f"Training Neural Network on {self.device}...")
        t0 = time.perf_counter()

        model = ClimateNN(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            n_classes=n_classes,
            dropout=cfg["dropout"],
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )
        criterion = nn.CrossEntropyLoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_vl = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val, dtype=torch.long).to(self.device)

        batch_size = cfg["batch_size"]
        n_epochs = cfg["epochs"]
        patience = cfg["early_stopping_patience"]

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= len(y_train)
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                val_out = model(X_vl)
                val_loss = criterion(val_out, y_vl).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            val_preds = model(X_vl).argmax(dim=1).cpu().numpy()

        training_time = time.perf_counter() - t0
        result = {
            "model": model,
            "training_time": training_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_preds": val_preds,
            "best_val_loss": best_val_loss,
        }
        logger.info(f"NN done in {training_time:.1f}s. Best val loss: {best_val_loss:.4f}")
        self.results["neural_network"] = result
        return result

    def train_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train all classical models and return combined results."""
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_svm(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_neural_network(X_train, y_train, X_val, y_val)
        return self.results

    def save_models(self, models_dir: Path) -> None:
        """Save all trained models to disk."""
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, result in self.results.items():
            model = result["model"]
            if name == "neural_network":
                torch.save(model.state_dict(),
                           models_dir / f"{name}_state_dict.pt")
                # Also save architecture info
                joblib.dump({
                    "input_dim": next(model.parameters()).shape[1]
                    if hasattr(next(model.parameters()), 'shape') else None,
                    "train_losses": result["train_losses"],
                    "val_losses": result["val_losses"],
                }, models_dir / f"{name}_meta.pkl")
            else:
                joblib.dump(model, models_dir / f"{name}.pkl")

        logger.info(f"Saved {len(self.results)} models to {models_dir}")

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Run inference for a named model."""
        result = self.results[model_name]
        model = result["model"]
        if model_name == "neural_network":
            model.eval()
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return model(X_t).argmax(dim=1).cpu().numpy()
        return model.predict(X)

    def scalability_analysis(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        sizes: Optional[list[int]] = None,
    ) -> "pd.DataFrame":
        """Train RF and XGBoost on increasing dataset sizes to show scalability."""
        import pandas as pd
        from sklearn.metrics import f1_score, accuracy_score

        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000, min(9000, len(X_train))]

        seed = self.config["data_split"]["random_state"]
        records = []

        for size in sizes:
            if size > len(X_train):
                continue
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=size, replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]

            for model_name, ModelClass, kwargs in [
                ("Random Forest",
                 RandomForestClassifier,
                 {"n_estimators": 100, "n_jobs": -1, "random_state": seed,
                  "class_weight": "balanced"}),
                ("XGBoost",
                 XGBClassifier,
                 {"n_estimators": 100, "tree_method": "hist",
                  "device": "cuda" if torch.cuda.is_available() else "cpu",
                  "verbosity": 0, "eval_metric": "mlogloss",
                  "random_state": seed}),
            ]:
                t0 = time.perf_counter()
                m = ModelClass(**kwargs)
                m.fit(X_sub, y_sub)
                train_time = time.perf_counter() - t0

                preds = m.predict(X_test)
                records.append({
                    "model": model_name,
                    "train_size": size,
                    "accuracy": accuracy_score(y_test, preds),
                    "f1_macro": f1_score(y_test, preds, average="macro",
                                         zero_division=0),
                    "training_time": train_time,
                })
                logger.info(f"Scalability [{model_name}, n={size}]: "
                            f"f1={records[-1]['f1_macro']:.3f}, "
                            f"t={train_time:.1f}s")

        return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    from src.features.engineering import load_splits
    splits, _ = load_splits(cfg)

    trainer = ClassicalModelTrainer(cfg)
    results = trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    trainer.save_models(Path(cfg["paths"]["models"]))
    print("Training complete.")
    for name, r in results.items():
        print(f"  {name}: {r['training_time']:.1f}s")
