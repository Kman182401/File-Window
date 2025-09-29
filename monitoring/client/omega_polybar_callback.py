import json
import math
import os
import tempfile
import time
from typing import Any, Callable, Dict, Optional

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover - optional dependency
    BaseCallback = object  # type: ignore[misc,assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


OUT_PATH = os.path.expanduser("~/.local/share/omega/learning_status.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def _write_atomic_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _safe(value: Any, ndigits: int = 6) -> Optional[float]:
    """Return a rounded float when possible, otherwise None."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except Exception:
        return None


class OmegaPolybarCallback(BaseCallback):
    """Callback that writes compact PPO telemetry for the Polybar formatter."""

    def __init__(
        self,
        write_every_steps: int = 2048,
        eval_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        verbose: int = 0,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Stable-Baselines3 must be installed to use OmegaPolybarCallback."
            ) from _IMPORT_ERROR
        super().__init__(verbose)
        self.write_every_steps = int(write_every_steps)
        self.eval_provider = eval_provider
        self._last_write_steps = 0

    def _on_step(self) -> bool:
        steps = int(self.model.num_timesteps)
        if steps - self._last_write_steps < self.write_every_steps:
            return True
        self._last_write_steps = steps

        log_dict = self.logger.get_log_dict()
        data: Dict[str, Any] = {
            "ts": int(time.time()),
            "algo": "ppo",
            "total_timesteps": steps,
            "metrics": {
                "approx_kl": _safe(log_dict.get("train/approx_kl")),
                "entropy": _safe(log_dict.get("train/entropy_loss")),
                "clipfrac": _safe(log_dict.get("train/clip_fraction")),
                "policy_loss": _safe(log_dict.get("train/policy_gradient_loss")),
                "value_loss": _safe(log_dict.get("train/value_loss")),
                "ep_rew_mean": _safe(log_dict.get("rollout/ep_rew_mean")),
                "fps": _safe(log_dict.get("time/fps")),
            },
        }

        if callable(self.eval_provider):
            try:
                data["eval"] = self.eval_provider()
            except Exception as exc:  # keep JSON valid even on failures
                data["eval"] = {"error": repr(exc)}

        _write_atomic_json(OUT_PATH, data)
        return True
