"""
Classical ML models for climate temperature regression.

Models:
  - XGBoost Regressor
  - Random Forest Regressor
  - Ridge Regression
  - Gradient Boosting Regressor
  - Neural Network Regressor (PyTorch, single output neuron, MSELoss)

All models expose a common interface: train_<model>() returns a result dict
with model, training_time, val_preds, and best_params (where applicable).
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PyTorch Neural Network — Regression
# ---------------------------------------------------------------------------

class ClimateRegressionNN(nn.Module):
    """Feedforward MLP for temperature regression.

    Output shape: (n, 1) for any input batch of shape (n, input_dim).
    Uses MSELoss; no softmax or classification head.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int],
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
        # Single output neuron for regression
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor of shape (n, 1)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Classical Regressor Trainer
# ---------------------------------------------------------------------------

class ClassicalRegressorTrainer:
    """Unified training interface for all classical regression models."""

    def __init__(self, config: Optional[dict] = None,
                 config_path: str = "config/config.yaml"):
        self.config = config or load_config(config_path)
        self.results: dict = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    def train_xgboost_regressor(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train XGBoost Regressor with GridSearchCV (neg_mean_squared_error)."""
        cfg = self.config["classical_ml"]["xgboost"]
        param_grid = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
        }
        logger.info("Training XGBoost Regressor with GridSearchCV...")
        t0 = time.perf_counter()

        tree_method = "hist"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        base = XGBRegressor(
            tree_method=tree_method,
            device=device_str,
            random_state=self.config["data_split"]["random_state"],
            verbosity=0,
        )
        gs = GridSearchCV(
            base, param_grid,
            cv=cfg["cv_folds"],
            scoring="neg_mean_squared_error",
            n_jobs=1,
            verbose=0,
        )
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
        logger.info(
            f"XGBoost Regressor done in {training_time:.1f}s. "
            f"Best params: {gs.best_params_}"
        )
        self.results["xgboost_regressor"] = result
        return result

    def train_random_forest_regressor(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train Random Forest Regressor with GridSearchCV (neg_mean_squared_error)."""
        cfg = self.config["classical_ml"]["random_forest"]
        param_grid = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_split": cfg["min_samples_split"],
        }
        logger.info("Training Random Forest Regressor with GridSearchCV...")
        t0 = time.perf_counter()

        base = RandomForestRegressor(
            random_state=self.config["data_split"]["random_state"],
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base, param_grid,
            cv=cfg["cv_folds"],
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
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
        logger.info(
            f"Random Forest Regressor done in {training_time:.1f}s. "
            f"Best params: {gs.best_params_}"
        )
        self.results["random_forest_regressor"] = result
        return result

    def train_ridge_regression(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train Ridge Regression with GridSearchCV over alpha values."""
        # Read alpha list from config; fall back to default if key absent
        ridge_cfg = self.config["classical_ml"].get("ridge", {})
        alpha_list = ridge_cfg.get("alpha", [0.01, 0.1, 1.0, 10.0, 100.0])
        cv_folds = ridge_cfg.get("cv_folds", 3)

        param_grid = {"alpha": alpha_list}
        logger.info("Training Ridge Regression with GridSearchCV...")
        t0 = time.perf_counter()

        base = Ridge(
            random_state=self.config["data_split"]["random_state"],
        )
        gs = GridSearchCV(
            base, param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
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
        logger.info(
            f"Ridge Regression done in {training_time:.1f}s. "
            f"Best params: {gs.best_params_}"
        )
        self.results["ridge_regression"] = result
        return result

    def train_gradient_boosting_regressor(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train Gradient Boosting Regressor with GridSearchCV (neg_mean_squared_error)."""
        cfg = self.config["classical_ml"]["gradient_boosting"]
        param_grid = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
        }
        logger.info("Training Gradient Boosting Regressor with GridSearchCV...")
        t0 = time.perf_counter()

        base = GradientBoostingRegressor(
            random_state=self.config["data_split"]["random_state"],
        )
        gs = GridSearchCV(
            base, param_grid,
            cv=cfg["cv_folds"],
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
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
        logger.info(
            f"Gradient Boosting Regressor done in {training_time:.1f}s. "
            f"Best params: {gs.best_params_}"
        )
        self.results["gradient_boosting_regressor"] = result
        return result

    def train_neural_network_regressor(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train ClimateRegressionNN (PyTorch MLP, MSELoss, single output neuron)."""
        cfg = self.config["classical_ml"]["neural_net"]
        input_dim = X_train.shape[1]

        logger.info(f"Training Neural Network Regressor on {self.device}...")
        t0 = time.perf_counter()

        model = ClimateRegressionNN(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_vl = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

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
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # squeeze (n, 1) -> (n,) for val_preds consistency
            val_preds = model(X_vl).squeeze(1).cpu().numpy()

        training_time = time.perf_counter() - t0
        result = {
            "model": model,
            "training_time": training_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_preds": val_preds,
            "best_val_loss": best_val_loss,
        }
        logger.info(
            f"Neural Network Regressor done in {training_time:.1f}s. "
            f"Best val MSE loss: {best_val_loss:.4f}"
        )
        self.results["neural_network_regressor"] = result
        return result

    def train_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        """Train all five regression models and return combined results dict."""
        self.train_xgboost_regressor(X_train, y_train, X_val, y_val)
        self.train_random_forest_regressor(X_train, y_train, X_val, y_val)
        self.train_ridge_regression(X_train, y_train, X_val, y_val)
        self.train_gradient_boosting_regressor(X_train, y_train, X_val, y_val)
        self.train_neural_network_regressor(X_train, y_train, X_val, y_val)
        return self.results

    def save_models(self, models_dir: Path) -> None:
        """Save all trained models to disk.

        sklearn models → <name>.pkl
        Neural network → neural_network_regressor_state_dict.pt
                       + neural_network_regressor_meta.pkl
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, result in self.results.items():
            model = result["model"]
            if name == "neural_network_regressor":
                torch.save(
                    model.state_dict(),
                    models_dir / "neural_network_regressor_state_dict.pt",
                )
                joblib.dump(
                    {
                        "hidden_dims": self.config["classical_ml"]["neural_net"]["hidden_dims"],
                        "dropout": self.config["classical_ml"]["neural_net"]["dropout"],
                        "train_losses": result["train_losses"],
                        "val_losses": result["val_losses"],
                        "best_val_loss": result["best_val_loss"],
                    },
                    models_dir / "neural_network_regressor_meta.pkl",
                )
            else:
                joblib.dump(model, models_dir / f"{name}.pkl")

        logger.info(f"Saved {len(self.results)} models to {models_dir}")

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Run inference for a named model; returns float array for regression."""
        result = self.results[model_name]
        model = result["model"]
        if model_name == "neural_network_regressor":
            model.eval()
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                # squeeze (n, 1) -> (n,)
                return model(X_t).squeeze(1).cpu().numpy()
        return model.predict(X)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    from src.features.engineering import load_splits
    splits, _ = load_splits(cfg)

    trainer = ClassicalRegressorTrainer(cfg)
    results = trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    trainer.save_models(Path(cfg["paths"]["models"]))
    print("Training complete.")
    for name, r in results.items():
        print(f"  {name}: {r['training_time']:.1f}s")
