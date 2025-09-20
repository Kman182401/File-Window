from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sequence_dataset import Standardizer


class SequenceForecaster(nn.Module):
    """LSTM-based multi-horizon forecaster for sliding-window features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizons: Sequence[int] = (1, 5, 15),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._init_forget_gate_bias()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(self.horizons)),
        )

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _init_forget_gate_bias(self) -> None:
        hidden = self.hidden_size
        for layer in range(self.num_layers):
            for name in (f"bias_ih_l{layer}", f"bias_hh_l{layer}"):
                bias = getattr(self.lstm, name)
                bias.data[hidden: 2 * hidden] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.head(h)


@dataclass
class SequenceForecasterArtifact:
    model: SequenceForecaster
    scaler: Standardizer
    feature_cols: List[str]
    horizons: Tuple[int, ...]
    n_steps: int
    metrics: Dict[str, float]
    training_args: Dict[str, float]
    artifact_version: int = 1


def _to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))


def _dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(_to_tensor(X), _to_tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _compute_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    mse = np.mean((pred - target) ** 2, axis=0)
    return np.sqrt(mse)


def train_model(
    model: SequenceForecaster,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    horizons: Sequence[int],
    *,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    patience: int = 6,
) -> Tuple[SequenceForecaster, Dict[str, List[float]], Dict[str, float]]:
    """Train forecaster with early stopping; returns best model and metrics."""

    if not isinstance(device, torch.device):
        device = torch.device(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    horizons = tuple(int(h) for h in horizons)
    criterion = nn.MSELoss(reduction="none")

    train_loader = _dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    train_std = np.std(y_train, axis=0)
    weights_np = 1.0 / (np.asarray(train_std, dtype=np.float32) + 1e-6)
    weights_np = np.where(np.isfinite(weights_np), weights_np, 1.0)
    weights_norm = weights_np / max(float(weights_np.sum()), 1e-6)
    weight_tensor = torch.from_numpy(weights_norm).to(device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss_matrix = criterion(preds, batch_y)
            loss = (loss_matrix * weight_tensor).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss_matrix = criterion(preds, batch_y)
                loss = (loss_matrix * weight_tensor).sum(dim=1).mean()
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_preds = evaluate(model, X_val, y_val, device)
    val_rmse = _compute_rmse(val_preds, y_val)

    metrics = {f"val_rmse_h{h}": float(val_rmse[i]) for i, h in enumerate(horizons)}
    metrics["val_loss"] = float(best_val)
    metrics["horizon_weights"] = weights_norm.tolist()

    return model, history, metrics


def evaluate(
    model: SequenceForecaster,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    loader = _dataloader(X, y, batch_size=512, shuffle=False)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds.append(outputs.cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_artifact(
    path: Path,
    *,
    model: SequenceForecaster,
    scaler: Standardizer,
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    n_steps: int,
    metrics: Dict[str, float],
    training_args: Dict[str, float],
    artifact_version: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "feature_cols": list(feature_cols),
        "horizons": list(int(h) for h in horizons),
        "n_steps": int(n_steps),
        "scaler": scaler.state_dict(),
        "metrics": metrics,
        "training_args": training_args,
        "artifact_version": int(artifact_version),
    }
    torch.save(state, path)


def load_artifact(path: Path, device: Optional[torch.device] = None) -> SequenceForecasterArtifact:
    payload = torch.load(path, map_location=device or "cpu")
    feature_cols = payload["feature_cols"]
    horizons = tuple(payload["horizons"])
    n_steps = int(payload["n_steps"])

    model = SequenceForecaster(
        input_size=len(feature_cols),
        horizons=horizons,
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()

    scaler = Standardizer()
    scaler.load_state_dict(payload["scaler"])

    return SequenceForecasterArtifact(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        horizons=horizons,
        n_steps=n_steps,
        metrics=payload.get("metrics", {}),
        training_args=payload.get("training_args", {}),
        artifact_version=int(payload.get("artifact_version", 1)),
    )
