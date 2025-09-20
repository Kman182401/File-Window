from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from ml.sequence_dataset import Standardizer, chrono_split, make_sequences
from ml.sequence_forecaster import (
    SequenceForecaster,
    evaluate,
    save_artifact,
    train_model,
)


def fit_seq_forecaster_for_symbol(
    df_feats: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    symbol: str,
    horizons: Iterable[int] = (1, 5, 15),
    n_steps: int = 150,
    step: int = 1,
    out_dir: str | Path = "models/seq_forecaster",
    train_split: float = 0.7,
    val_split: float = 0.15,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    patience: int = 6,
    device: str | torch.device | None = None,
    seed: int | None = 42,
) -> Dict[str, float]:
    """Fit a sequence forecaster on a single symbol and persist the artifact.

    Parameters mirror the CLI defaults so this helper can be invoked directly
    from pipeline or notebook jobs.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if "timestamp" in df_feats.columns:
        df_feats = df_feats.sort_values("timestamp").reset_index(drop=True)

    horizons_tuple = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not horizons_tuple:
        raise ValueError("At least one positive horizon is required.")

    X, y_dict = make_sequences(
        df_feats,
        list(feature_cols),
        horizons=horizons_tuple,
        n_steps=n_steps,
        step=step,
    )

    if X.shape[0] < 3:
        raise ValueError("Not enough windows to split into train/val/test sets.")

    idx_tr, idx_va, idx_te = chrono_split(X.shape[0], train=train_split, val=val_split)
    if len(idx_va) == 0 or len(idx_te) == 0:
        raise ValueError("Validation/test splits are empty. Adjust train/val fractions.")

    scaler = Standardizer()
    scaler.fit(X[idx_tr])

    X_tr = scaler.transform(X[idx_tr])
    X_va = scaler.transform(X[idx_va])
    X_te = scaler.transform(X[idx_te])

    X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
    X_va = np.ascontiguousarray(X_va, dtype=np.float32)
    X_te = np.ascontiguousarray(X_te, dtype=np.float32)

    if set(y_dict.keys()) != set(horizons_tuple):
        raise ValueError("Horizon mismatch between requested horizons and generated labels")

    if any(len(y_dict[h]) != X.shape[0] for h in horizons_tuple):
        raise ValueError("Label lengths do not match window count")

    y_matrix = np.stack([y_dict[h] for h in horizons_tuple], axis=1)
    y_tr = np.ascontiguousarray(y_matrix[idx_tr], dtype=np.float32)
    y_va = np.ascontiguousarray(y_matrix[idx_va], dtype=np.float32)
    y_te = np.ascontiguousarray(y_matrix[idx_te], dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model = SequenceForecaster(
        input_size=X.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizons=horizons_tuple,
        dropout=dropout,
    )

    model, history, metrics = train_model(
        model,
        X_tr,
        y_tr,
        X_va,
        y_va,
        horizons_tuple,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    test_preds = evaluate(model, X_te, y_te, device)
    test_mse = np.mean((test_preds - y_te) ** 2, axis=0)
    test_rmse = np.sqrt(test_mse)

    metrics.update({f"test_rmse_h{h}": float(test_rmse[i]) for i, h in enumerate(horizons_tuple)})
    metrics["test_loss"] = float(np.mean(test_mse))
    metrics.update(
        {
            "train_samples": int(len(idx_tr)),
            "val_samples": int(len(idx_va)),
            "test_samples": int(len(idx_te)),
        }
    )

    artifact_dir = Path(out_dir) / symbol
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"

    training_args = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "device": str(device),
        "step": step,
        "train_split": train_split,
        "val_split": val_split,
        "symbol": symbol,
        "seed": seed,
    }

    save_artifact(
        model_path,
        model=model,
        scaler=scaler,
        feature_cols=list(feature_cols),
        horizons=horizons_tuple,
        n_steps=n_steps,
        metrics=metrics,
        training_args=training_args,
        artifact_version=1,
    )

    # Persist diagnostic files alongside the artifact for quick inspection.
    feat_hash = hashlib.sha1("|".join(feature_cols).encode("utf-8")).hexdigest()

    sample_positions = (np.arange(X.shape[0]) * step) + (n_steps - 1)
    train_end_index = int(sample_positions[idx_tr[-1]]) if len(idx_tr) else None

    train_end_ts = None
    if train_end_index is not None and "timestamp" in df_feats.columns:
        timestamp_series = pd.to_datetime(df_feats["timestamp"], errors="coerce")
        base_frame = df_feats[feature_cols + ["close"]].dropna()
        timestamp_series = timestamp_series.loc[base_frame.index]
        max_h = max(horizons_tuple)
        if max_h > 0 and len(base_frame) > max_h:
            base_frame = base_frame.iloc[:-max_h]
            timestamp_series = timestamp_series.iloc[:-max_h]
        base_frame = base_frame.reset_index(drop=True)
        timestamp_series = timestamp_series.reset_index(drop=True)
        if 0 <= train_end_index < len(timestamp_series):
            ts_value = timestamp_series.iloc[train_end_index]
            train_end_ts = ts_value.isoformat() if isinstance(ts_value, pd.Timestamp) else ts_value

    metrics_with_meta = metrics | {
        "feature_hash": feat_hash,
        "train_end_index": train_end_index,
        "train_end_ts": train_end_ts,
    }

    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_with_meta, f, indent=2)
    with open(artifact_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(artifact_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": list(feature_cols),
                "horizons": list(horizons_tuple),
                "n_steps": n_steps,
                "step": step,
                "symbol": symbol,
            },
            f,
            indent=2,
        )

    return metrics_with_meta
