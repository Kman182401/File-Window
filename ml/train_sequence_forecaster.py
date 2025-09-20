from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from .sequence_dataset import Standardizer, chrono_split, make_sequences
from .sequence_forecaster import SequenceForecaster, evaluate, save_artifact, train_model


def _infer_feature_cols(df: pd.DataFrame, provided: Sequence[str] | None) -> List[str]:
    if provided:
        return list(provided)
    skip = {"timestamp", "datetime", "date", "open", "high", "low", "close", "volume"}
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    features = [c for c in numeric_cols if c not in skip]
    if not features:
        raise ValueError("No numeric feature columns detected; supply --feature-cols explicitly.")
    return features


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence forecaster for short-horizon returns.")
    parser.add_argument("data", type=Path, help="Path to feature dataframe (parquet or CSV).")
    parser.add_argument("symbol", type=str, help="Symbol identifier for artifact storage.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/seq_forecaster"))
    parser.add_argument("--feature-cols", nargs="*", default=None, help="Explicit feature columns to use.")
    parser.add_argument("--n-steps", type=int, default=150, help="Sequence length (timesteps).")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 15], help="Prediction horizons (bars ahead).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--step", type=int, default=1, help="Stride between windows to reduce overlap.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training share of samples.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation share of samples.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_frame(args.data)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates().reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("Input dataframe must include a 'close' column for return targets.")

    feature_cols = _infer_feature_cols(df, args.feature_cols)
    horizons = tuple(sorted(int(h) for h in args.horizons))

    X, y_dict = make_sequences(df, feature_cols, horizons=horizons, n_steps=args.n_steps, step=args.step)
    y_matrix = np.stack([y_dict[h] for h in horizons], axis=1)

    idx_train, idx_val, idx_test = chrono_split(len(X), train=args.train_split, val=args.val_split)
    if len(idx_test) == 0:
        raise ValueError("Test split is empty; adjust --train-split/--val-split.")

    scaler = Standardizer()
    scaler.fit(X[idx_train])
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_test = scaler.transform(X[idx_test])

    y_train = y_matrix[idx_train]
    y_val = y_matrix[idx_val]
    y_test = y_matrix[idx_test]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequenceForecaster(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        horizons=horizons,
        dropout=args.dropout,
    )

    model, history, metrics = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        horizons,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    test_preds = evaluate(model, X_test, y_test, device=device)
    test_mse = np.mean((test_preds - y_test) ** 2, axis=0)
    test_rmse = np.sqrt(test_mse)

    for i, h in enumerate(horizons):
        metrics[f"test_rmse_h{h}"] = float(test_rmse[i])
    metrics["test_loss"] = float(np.mean(test_mse))
    metrics.update(
        {
            "train_samples": int(len(idx_train)),
            "val_samples": int(len(idx_val)),
            "test_samples": int(len(idx_test)),
        }
    )

    artifact_dir = args.output_dir / args.symbol
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"

    training_args = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "num_layers": args.layers,
        "device": str(device),
        "step": args.step,
    }

    save_artifact(
        model_path,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        horizons=horizons,
        n_steps=args.n_steps,
        metrics=metrics,
        training_args=training_args,
    )

    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(artifact_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(artifact_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "horizons": list(horizons), "n_steps": args.n_steps}, f, indent=2)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
