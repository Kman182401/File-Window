from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - optional heavy deps
    import torch  # noqa: F401
    from stable_baselines3 import SAC  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from wfo_rl import RLAdapter, RLSpec, make_env_from_df


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch/stable-baselines3 not available (pip install gymnasium stable-baselines3 sb3-contrib)",
)
def test_rl_smoke(tmp_path):
    bars = 390
    idx = pd.date_range(end=datetime.now(UTC), periods=bars, freq="1min")
    prices = 100 + np.cumsum(np.random.randn(bars) * 0.05)
    df = pd.DataFrame({
        "timestamp": idx,
        "close": prices,
        "volume": np.random.randint(50, 500, size=bars),
    })
    df["returns"] = df["close"].pct_change().fillna(0.0)

    train_df = df.iloc[:300].reset_index(drop=True)
    test_df = df.iloc[300:].reset_index(drop=True)

    spec = RLSpec(
        algo="SAC",
        policy="MlpPolicy",
        train_timesteps=500,
        n_envs=1,
        seed=7,
        vecnormalize_obs=True,
        vecnormalize_reward=True,
    )
    adapter = RLAdapter(spec, fast_smoke=True)
    reward_kwargs = {"lambda_var": 0.1, "lambda_dd": 0.05, "h_var": 30}
    train_env = make_env_from_df(train_df, costs_bps=0.2, reward_kwargs=reward_kwargs, eval_mode=False)
    oos_env = make_env_from_df(test_df, costs_bps=0.2, reward_kwargs=reward_kwargs, eval_mode=True)

    train_instance = train_env()
    eval_instance = oos_env()

    assert not train_instance.eval_mode
    assert eval_instance.eval_mode
    assert not eval_instance.use_hindsight_in_training

    train_instance.close()
    eval_instance.close()

    model, vecnorm_path = adapter.fit_on_is(train_env, str(tmp_path))
    returns = adapter.score_on_oos(model, oos_env, vecnorm_path)

    assert returns.size > 0
    assert np.isfinite(returns).all()

    if hasattr(model, "save"):
        model.save(str(tmp_path / "policy.zip"))
        assert (tmp_path / "policy.zip").exists()
    if vecnorm_path:
        assert Path(vecnorm_path).exists()
