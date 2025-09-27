import numpy as np
import pandas as pd
import pytest


def _sb3_available() -> bool:
    try:  # pragma: no cover - runtime check
        import torch  # noqa: F401
        import stable_baselines3  # noqa: F401
        from sb3_contrib import RecurrentPPO  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sb3_available(), reason="SB3/torch unavailable")
def test_rl_smoke(monkeypatch, tmp_path):
    from datetime import datetime
    from wfo import runner as wfo_runner

    # Synthetic one-day minute data with returns
    periods = 6 * 60
    idx = pd.date_range(datetime(2024, 1, 1), periods=periods, freq="1min", tz="America/New_York")
    df = pd.DataFrame({
        "timestamp": idx,
        "returns": np.random.normal(0, 1e-4, size=periods),
    })

    monkeypatch.setattr(
        "wfo.data_access.MarketDataAccess.get_bars",
        lambda self, *args, **kwargs: df,
    )

    captured = {}

    class DummyAdapter:
        def __init__(self, spec, fast_smoke):
            captured["spec_algo"] = spec.algo
            captured["fast"] = fast_smoke

        def fit_on_is(self, make_env_fn, log_dir):
            captured["is_len"] = getattr(make_env_fn, "_len", None)
            return object(), None

        def score_on_oos(self, model, make_env_fn, vecnorm):
            length = getattr(make_env_fn, "_len", 10)
            captured["oos_len"] = length
            return np.full(length, 1e-4)

    def fake_make_env(df_slice, costs_bps=0.0):
        fn = lambda: {"df": df_slice}
        setattr(fn, "_len", len(df_slice))
        return fn

    monkeypatch.setattr("wfo.runner.RLAdapter", DummyAdapter)
    monkeypatch.setattr("wfo.runner.make_env_from_df", fake_make_env)

    out_dir = tmp_path / "artifacts"
    result = wfo_runner.run_wfo(
        symbols=["ES"],
        is_days=1,
        oos_days=0,
        step_days=1,
        cycles_min=1,
        embargo_days=0,
        label_lookahead_bars=0,
        cpcv_folds=2,
        config_path=None,
        dry_run=True,
        output_root=out_dir,
        strategies=[
            {
                "name": "ES_PPO_LSTM",
                "type": "rl_policy",
                "algo": "RecurrentPPO",
                "policy": "MlpLstmPolicy",
                "rl": {"train_timesteps": 200, "n_envs": 1},
            }
        ],
        rl_fast_smoke=True,
    )

    artifacts = list(out_dir.glob("*"))
    assert artifacts, "Expected RL artifacts"
    assert captured.get("spec_algo") == "RecurrentPPO"
    assert captured.get("is_len") == len(df)
    assert captured.get("oos_len") == len(df)
    assert result["summary"]["sharpe_mean"] != 0
