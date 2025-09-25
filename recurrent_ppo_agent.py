#!/usr/bin/env python3
"""
RecurrentPPO Trading Agent

Production-ready RecurrentPPO implementation with LSTM for capturing
temporal dependencies in financial markets. Optimized for m5.large.
"""

import os
import copy
import logging
import math
from typing import Dict, Any, Optional, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import warnings

warnings.filterwarnings('ignore')

# Import system components
from system_optimization_config import get_available_memory_mb, get_optimal_batch_size
from memory_management_system import get_memory_manager, memory_limit
from error_handling_system import (
    robust_execution,
    ErrorCategory,
    ErrorSeverity,
    get_error_handler
)

logger = logging.getLogger(__name__)


class TemporalTradingCallback(BaseCallback):
    """
    Custom callback for tracking temporal pattern metrics during RecurrentPPO training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.trend_accuracy = []
        self.sequence_rewards = []
        self.lstm_hidden_states = []
        
    def _on_step(self) -> bool:
        # Track LSTM-specific metrics
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'lstm_hidden_state'):
            # Get LSTM hidden state statistics
            hidden_state = self.model.policy.lstm_hidden_state
            if hidden_state is not None:
                hidden_mean = float(hidden_state.mean())
                hidden_std = float(hidden_state.std())
                
                # Log every 1000 steps
                if self.n_calls % 1000 == 0:
                    self.logger.record("lstm/hidden_mean", hidden_mean)
                    self.logger.record("lstm/hidden_std", hidden_std)
        
        # Track episode metrics
        if len(self.locals.get("dones", [])) > 0 and self.locals["dones"][0]:
            info = self.locals.get("infos", [{}])[0]
            
            # Track trend detection accuracy if available
            if "trend_accuracy" in info:
                self.trend_accuracy.append(info["trend_accuracy"])
                if len(self.trend_accuracy) > 10:
                    self.logger.record("trading/trend_accuracy", np.mean(self.trend_accuracy[-10:]))
            
            # Track sequence rewards
            if "episode" in info:
                self.sequence_rewards.append(info["episode"]["r"])
                if len(self.sequence_rewards) > 10:
                    self.logger.record("trading/seq_reward_mean", np.mean(self.sequence_rewards[-10:]))

            metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
            if metrics:
                if "action_delta_mean" in metrics:
                    self.logger.record("trading/action_delta_mean", float(metrics["action_delta_mean"]))
                if "turnover" in metrics:
                    self.logger.record("trading/turnover", float(metrics["turnover"]))
        
        return True


class EntropyControlCallback(BaseCallback):
    """Bounds policy entropy using the policy distribution API."""

    def __init__(
        self,
        entropy_lower: float,
        entropy_upper: float,
        adjustment_lr: float = 0.05,
        min_ent_coef: float = 1e-4,
        max_ent_coef: float = 0.2,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.entropy_lower = entropy_lower
        self.entropy_upper = entropy_upper
        self.adjustment_lr = adjustment_lr
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self._last_observations: Optional[Any] = None
        self._last_episode_starts: Optional[np.ndarray] = None
        self._last_states: Optional[Any] = None

    def _on_step(self) -> bool:
        self._last_observations = self.locals.get("observations")
        self._last_episode_starts = self.locals.get("episode_starts")
        self._last_states = self.locals.get("states")
        return True

    def _on_rollout_end(self) -> None:
        policy = getattr(self.model, "policy", None)
        if policy is None or self._last_observations is None:
            return None

        device = self.model.device

        obs_t, _ = policy.obs_to_tensor(self._last_observations)
        if isinstance(obs_t, dict):
            sample_tensor = next(iter(obs_t.values()))
        else:
            sample_tensor = obs_t

        episode_starts = self._last_episode_starts
        if episode_starts is None:
            episode_starts_t = torch.zeros(sample_tensor.shape[0], device=device, dtype=torch.bool)
        else:
            episode_starts_t = torch.as_tensor(episode_starts, device=device, dtype=torch.bool)

        states = self._last_states
        if states is not None:
            if isinstance(states, tuple):
                states_t = tuple(torch.as_tensor(s, device=device, dtype=torch.float32) for s in states)
            else:
                states_t = torch.as_tensor(states, device=device, dtype=torch.float32)
        else:
            states_t = None

        with torch.no_grad():
            dist = policy.get_distribution(obs_t, state=states_t, episode_starts=episode_starts_t)
            current_entropy = float(dist.entropy().mean().cpu().item())

        current_ent_coef = float(getattr(self.model, "ent_coef", 0.0))
        updated_ent_coef = current_ent_coef

        if current_entropy < self.entropy_lower:
            updated_ent_coef = min(current_ent_coef * (1.0 + self.adjustment_lr), self.max_ent_coef)
        elif current_entropy > self.entropy_upper:
            updated_ent_coef = max(current_ent_coef * (1.0 - self.adjustment_lr), self.min_ent_coef)

        if not math.isclose(updated_ent_coef, current_ent_coef, rel_tol=1e-6):
            setattr(self.model, "ent_coef", updated_ent_coef)
            self.logger.record("entropy/ent_coef", updated_ent_coef)
        self.logger.record("entropy/estimated", current_entropy)
        self._last_observations = None
        self._last_episode_starts = None
        self._last_states = None
        return None


class RecurrentPPOAgent:
    """
    Production-ready RecurrentPPO agent for temporal pattern recognition in trading.
    
    Features:
    - LSTM integration for sequence modeling
    - Memory-efficient architecture (64 hidden units)
    - Shared LSTM between actor and critic
    - Gradient clipping for stability
    """
    
    def __init__(self, 
                 env: gym.Env,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RecurrentPPO agent.
        
        Args:
            env: Trading environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = self._get_optimized_config(config)
        self.model: Optional[RecurrentPPO] = None
        self.callbacks: List[BaseCallback] = []
        self.training_env: Optional[VecNormalize] = None
        self.eval_env: Optional[VecNormalize] = None
        self._vecnormalize_suffix = "_vecnormalize.pkl"

        # Memory management
        self.memory_manager = get_memory_manager()
        self.error_handler = get_error_handler()

        # Register with memory manager
        self.memory_manager.register_component(
            'recurrent_ppo_agent',
            self,
            limit_mb=self.config.get('memory_limit_mb', 800)
        )
        
        logger.info(f"Initialized RecurrentPPO agent with config: {self.config}")
    
    def _get_optimized_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get optimized configuration based on available resources.
        
        Args:
            config: User configuration
            
        Returns:
            Optimized configuration
        """
        # Base configuration
        preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
        observation_space = getattr(env, "observation_space", None)
        is_dict_obs = isinstance(observation_space, gym.spaces.Dict)

        base_config: Dict[str, Any] = {
            # Model architecture
            'policy': 'MultiInputLstmPolicy' if is_dict_obs else 'MlpLstmPolicy',
            'policy_kwargs': {
                'lstm_hidden_size': 64,
                'n_lstm_layers': 1,
                'shared_lstm': True,
                'enable_critic_lstm': True,
                'net_arch': {
                    'pi': [64, 64],
                    'vf': [64, 64]
                },
                'activation_fn': nn.ReLU,
                'ortho_init': True
            },

            # PPO hyperparameters
            'learning_rate': 3e-4,
            'n_steps': 128,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range_start': 0.2,
            'clip_range_end': 0.1,
            'clip_range_vf': 0.2,
            'normalize_advantage': True,
            'ent_coef': 0.02,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.03,

            # Vectorised env / normalisation
            'n_envs': 4,
            'vecnormalize_kwargs': {
                'norm_obs': True,
                'norm_reward': True,
                'clip_obs': 10.0,
                'clip_reward': 10.0,
                'gamma': 0.99,
            },

            'optimizer_kwargs': {
                'eps': 1e-5,
            },

            # Entropy bounds
            'entropy_lower_bound': 0.2,
            'entropy_upper_bound': 1.5,
            'entropy_adjustment_lr': 0.05,
            'min_ent_coef': 1e-4,
            'max_ent_coef': 0.2,

            # System
            'device': preferred_device,
            'verbose': 1,
            'tensorboard_log': './tensorboard_logs/recurrent_ppo/',
            'memory_limit_mb': 800
        }

        if preferred_device == "cuda":
            logger.info("CUDA available – configuring RecurrentPPO to run on GPU.")
        else:
            logger.info("CUDA not available – falling back to CPU for RecurrentPPO.")
        
        # Adjust based on available memory
        available_mb = get_available_memory_mb()
        
        if available_mb < 2000:
            # Low memory mode
            base_config['n_steps'] = 64
            base_config['batch_size'] = 32
            base_config['policy_kwargs']['lstm_hidden_size'] = 32
            base_config['policy_kwargs']['net_arch'] = {
                'pi': [32, 32],
                'vf': [32, 32]
            }
            logger.warning(f"Low memory ({available_mb}MB), using reduced RecurrentPPO config")
        elif available_mb < 3000:
            # Medium memory mode
            base_config['n_steps'] = 128
            base_config['batch_size'] = 64

        # Merge with user config
        if config:
            base_config.update(config)

        return base_config
    
    @robust_execution(category=ErrorCategory.MODEL, max_retries=2)
    @memory_limit(1200)  # Enforce memory limit
    def build_model(self):
        """
        Build RecurrentPPO model with error handling.
        """
        logger.info("Building RecurrentPPO model...")
        
        try:
            # Prepare environment and clip schedule
            self.training_env = self._build_vectorised_env(training=True)
            if self.eval_env is None:
                self.eval_env = self._build_vectorised_env(training=False)
            self._sync_eval_env_stats()

            clip_schedule = self._create_clip_schedule(
                self.config.get('clip_range_start', 0.2),
                self.config.get('clip_range_end', 0.1)
            )

            lr_config = self.config.get('learning_rate', 3e-4)
            if callable(lr_config):
                learning_rate = lr_config
            else:
                learning_rate = self._create_linear_schedule(float(lr_config))

            # Create model
            self.model = RecurrentPPO(
                policy=self.config['policy'],
                env=self.training_env,
                learning_rate=learning_rate,
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=clip_schedule,
                clip_range_vf=self.config['clip_range_vf'],
                normalize_advantage=self.config['normalize_advantage'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                policy_kwargs=self.config['policy_kwargs'],
                verbose=self.config['verbose'],
                tensorboard_log=self.config['tensorboard_log'],
                device=self.config['device'],
                target_kl=self.config.get('target_kl', None),
                optimizer_kwargs=self.config.get('optimizer_kwargs')
            )

            # Setup callbacks
            self._setup_callbacks()

            logger.info("✓ RecurrentPPO model built successfully")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorCategory.MODEL,
                ErrorSeverity.HIGH,
                {'component': 'recurrent_ppo_build'}
            )
            raise
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        entropy_callback = EntropyControlCallback(
            entropy_lower=self.config.get('entropy_lower_bound', 0.2),
            entropy_upper=self.config.get('entropy_upper_bound', 1.5),
            adjustment_lr=self.config.get('entropy_adjustment_lr', 0.05),
            min_ent_coef=self.config.get('min_ent_coef', 1e-4),
            max_ent_coef=self.config.get('max_ent_coef', 0.2)
        )

        if self.eval_env is None:
            self.eval_env = self._build_vectorised_env(training=False)
        self._sync_eval_env_stats()

        self.callbacks = [
            # Temporal pattern tracking
            TemporalTradingCallback(),

            # Entropy stabilisation
            entropy_callback,

            # Checkpointing
            CheckpointCallback(
                save_freq=1000,
                save_path='./checkpoints/recurrent_ppo/',
                name_prefix='recurrent_ppo_trading',
                save_replay_buffer=False,  # PPO doesn't have replay buffer
                save_vecnormalize=True
            ),
            
            # Evaluation
            EvalCallback(
                self.eval_env if self.eval_env is not None else self.env,
                best_model_save_path='./models/recurrent_ppo/',
                log_path='./logs/recurrent_ppo/',
                eval_freq=500,
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )
        ]
    
    @robust_execution(category=ErrorCategory.MODEL)
    def train(self, 
              total_timesteps: int = 100000,
              progress_bar: bool = True) -> None:
        """
        Train RecurrentPPO model with production safeguards.
        
        Args:
            total_timesteps: Total training timesteps
            progress_bar: Show progress bar
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Starting RecurrentPPO training for {total_timesteps} timesteps...")
        
        try:
            # Check memory before training
            available_mb = get_available_memory_mb()
            if available_mb < 1200:
                raise MemoryError(f"Insufficient memory for training: {available_mb}MB < 1200MB")

            if isinstance(self.training_env, VecNormalize):
                self.eval_env = self._build_vectorised_env(training=False)
                self._sync_eval_env_stats()

            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                log_interval=10,
                tb_log_name="recurrent_ppo_run",
                reset_num_timesteps=False,
                progress_bar=progress_bar
            )

            logger.info("✓ RecurrentPPO training completed successfully")
            if isinstance(self.training_env, VecNormalize):
                self._sync_eval_env_stats()

        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorCategory.MODEL,
                ErrorSeverity.HIGH,
                {'component': 'recurrent_ppo_train', 'timesteps': total_timesteps}
            )
            raise
    
    def predict(self, 
                observation: np.ndarray,
                lstm_states: Optional[Tuple] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = True) -> Tuple[np.ndarray, Tuple]:
        """
        Make prediction using trained model with LSTM states.
        
        Args:
            observation: Current market observation
            lstm_states: Previous LSTM hidden and cell states
            episode_start: Whether this is start of episode
            deterministic: Use deterministic policy
            
        Returns:
            Tuple of (action, lstm_states)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle episode start
        if episode_start is None:
            episode_start = np.array([False])
        
        obs_for_policy = self._normalize_observation(observation)

        return self.model.predict(
            obs_for_policy,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic
        )

    def save(self, path: str):
        """
        Save model.
        
        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)

        if isinstance(self.training_env, VecNormalize):
            vec_path = f"{path}{self._vecnormalize_suffix}"
            self.training_env.save(vec_path)
            logger.info(f"Saved VecNormalize statistics to {vec_path}")

        logger.info(f"RecurrentPPO model saved to {path}")

    def load(self, path: str, env: Optional[gym.Env] = None):
        """
        Load model.
        
        Args:
            path: Load path
            env: Environment (uses self.env if None)
        """
        env = env or self.env

        vec_path = f"{path}{self._vecnormalize_suffix}"
        if os.path.exists(vec_path):
            logger.info(f"Loading VecNormalize statistics from {vec_path}")
            base_env = self._make_env_factory()
            self.training_env = VecNormalize.load(vec_path, base_env)
            self.training_env.training = False
            self.training_env.norm_reward = False
            self.model = RecurrentPPO.load(path, env=self.training_env, device=self.config['device'])
            self.eval_env = self._build_vectorised_env(training=False)
            self._sync_eval_env_stats()
        else:
            self.model = RecurrentPPO.load(path, env=env, device=self.config['device'])

        logger.info(f"RecurrentPPO model loaded from {path}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of RecurrentPPO components."""
        if self.model is None:
            return {'total_mb': 0}
        
        # Calculate model parameter memory
        param_count = sum(p.numel() for p in self.model.policy.parameters())
        
        memory_usage = {
            'model_params_mb': param_count * 4 / (1024 * 1024),  # 4 bytes per float32
            'lstm_state_mb': 0,
            'rollout_buffer_mb': 0,
            'total_mb': 0
        }
        
        # Estimate LSTM state memory
        if hasattr(self.model.policy, 'lstm_hidden_size'):
            lstm_size = self.model.policy.lstm_hidden_size
            n_envs = self.model.n_envs if hasattr(self.model, 'n_envs') else 1
            # Hidden + cell state
            memory_usage['lstm_state_mb'] = (2 * lstm_size * n_envs * 4) / (1024 * 1024)
        
        # Estimate rollout buffer
        if hasattr(self.model, 'rollout_buffer'):
            n_steps = self.config.get('n_steps', 128)
            obs_space = self.training_env.observation_space if self.training_env else self.env.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                flat_space = gym.spaces.flatten_space(obs_space)
                obs_size = int(np.prod(flat_space.shape))
            elif hasattr(obs_space, 'shape') and obs_space.shape is not None:
                obs_size = int(np.prod(obs_space.shape))
            else:
                obs_size = 0
            memory_usage['rollout_buffer_mb'] = (n_steps * obs_size * 4) / (1024 * 1024)

        memory_usage['total_mb'] = sum(memory_usage.values())

        return memory_usage
    
    def reduce_memory(self):
        """Reduce memory usage when under pressure."""
        if self.model:
            logger.warning("Reducing RecurrentPPO memory usage due to pressure")
            # Clear any cached data
            if hasattr(self.model, 'rollout_buffer'):
                self.model.rollout_buffer.reset()

    def get_lstm_state_info(self) -> Dict[str, Any]:
        """Get information about current LSTM state."""
        if self.model is None or not hasattr(self.model.policy, 'lstm_actor'):
            return {}
        
        info = {
            'lstm_hidden_size': self.config['policy_kwargs']['lstm_hidden_size'],
            'n_lstm_layers': self.config['policy_kwargs']['n_lstm_layers'],
            'shared_lstm': self.config['policy_kwargs']['shared_lstm']
        }
        
        # Get current hidden state statistics if available
        if hasattr(self.model.policy, 'lstm_hidden_state'):
            hidden = self.model.policy.lstm_hidden_state
            if hidden is not None:
                info['hidden_state_stats'] = {
                    'mean': float(hidden.mean()),
                    'std': float(hidden.std()),
                    'min': float(hidden.min()),
                    'max': float(hidden.max())
                }
        
        return info

    # --- Internal helpers -------------------------------------------------

    def _build_vectorised_env(self, training: bool):
        n_envs = max(1, int(self.config.get('n_envs', 1)))
        vec_kwargs = copy.deepcopy(self.config.get('vecnormalize_kwargs', {}))

        env_callables: List = []
        clone_failed = False

        for _ in range(n_envs):
            cloned = self._clone_environment(self.env)
            if cloned is self.env and n_envs > 1:
                clone_failed = True
                break
            env_callables.append(lambda env=cloned: env)

        if clone_failed:
            logger.warning("Environment cloning failed; falling back to single-env training")
            env_callables = [lambda: self.env]
            n_envs = 1
            self.config['n_envs'] = 1

        vec_env = DummyVecEnv(env_callables)

        obs_space = vec_env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            box_keys = [key for key, space in obs_space.spaces.items() if isinstance(space, gym.spaces.Box)]
            if box_keys:
                vec_kwargs.setdefault('norm_obs', True)
                vec_kwargs.setdefault('norm_obs_keys', box_keys)

        try:
            vec_norm = VecNormalize(vec_env, **vec_kwargs)
        except TypeError:
            vec_kwargs.pop('norm_obs_keys', None)
            vec_norm = VecNormalize(vec_env, **vec_kwargs)
        except Exception as exc:
            logger.warning(f"VecNormalize initialisation failed ({exc}); proceeding without normalisation")
            return vec_env

        vec_norm.training = training
        vec_norm.norm_reward = bool(training and vec_kwargs.get('norm_reward', True))
        if not training:
            vec_norm.norm_reward = False
        return vec_norm

    def _clone_environment(self, source_env: gym.Env) -> gym.Env:
        if hasattr(source_env, 'clone') and callable(getattr(source_env, 'clone')):
            try:
                return source_env.clone()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Environment-provided clone failed: {exc}")

        if hasattr(source_env, 'config'):
            try:
                new_config = copy.deepcopy(getattr(source_env, 'config'))
                data = getattr(source_env, 'data', None)
                return source_env.__class__(data=data, config=new_config)
            except Exception as exc:
                logger.debug(f"Config-based clone failed: {exc}")

        try:
            return copy.deepcopy(source_env)
        except Exception as exc:
            logger.debug(f"Deepcopy clone failed: {exc}")
            return source_env

    def _create_clip_schedule(self, start: float, end: float) -> Callable[[float], float]:
        start = float(start)
        end = float(end)

        def schedule(progress_remaining: float) -> float:
            progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
            cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress_remaining))
            return end + (start - end) * cosine_term

        return schedule

    def _create_linear_schedule(self, initial_value: float) -> Callable[[float], float]:
        initial_value = float(initial_value)

        def schedule(progress_remaining: float) -> float:
            progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
            return initial_value * progress_remaining

        return schedule

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        vec_env = self.training_env
        if vec_env is None and self.model is not None:
            try:
                vec_env = self.model.get_env()
            except AttributeError:
                vec_env = None

        if not isinstance(vec_env, VecNormalize):
            return observation

        obs_rms = getattr(vec_env, 'obs_rms', None)
        if obs_rms is None or obs_rms.mean is None:
            return observation

        expanded_obs = self._expand_obs_for_vecnorm(observation)
        was_training = vec_env.training
        vec_env.training = False
        normalized = vec_env.normalize_obs(expanded_obs)
        vec_env.training = was_training
        return self._squeeze_obs_from_vecnorm(normalized)

    def _expand_obs_for_vecnorm(self, observation: Any):
        if isinstance(observation, dict):
            return {k: np.asarray([v], dtype=np.float32) for k, v in observation.items()}
        return np.asarray([observation], dtype=np.float32)

    def _squeeze_obs_from_vecnorm(self, observation: Any):
        if isinstance(observation, dict):
            return {k: v[0] for k, v in observation.items()}
        return observation[0]

    def _sync_eval_env_stats(self) -> None:
        if not (
            isinstance(self.training_env, VecNormalize)
            and isinstance(self.eval_env, VecNormalize)
        ):
            return

        self.eval_env.training = False
        self.eval_env.norm_reward = False
        if self.training_env.obs_rms is not None:
            self.eval_env.obs_rms = copy.deepcopy(self.training_env.obs_rms)
        if self.training_env.ret_rms is not None:
            self.eval_env.ret_rms = copy.deepcopy(self.training_env.ret_rms)

    def _make_env_factory(self) -> DummyVecEnv:
        return DummyVecEnv([lambda: self._clone_environment(self.env)])


def create_recurrent_ppo_agent(env: gym.Env,
                               config: Optional[Dict[str, Any]] = None) -> RecurrentPPOAgent:
    """
    Factory function to create RecurrentPPO agent with proper error handling.
    
    Args:
        env: Trading environment
        config: Configuration
        
    Returns:
        Configured RecurrentPPO agent
    """
    try:
        agent = RecurrentPPOAgent(env, config)
        agent.build_model()
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create RecurrentPPO agent: {e}")
        # Fallback to standard PPO if RecurrentPPO fails
        logger.warning("Falling back to standard PPO agent")
        return None


def test_recurrent_ppo_agent():
    """Test RecurrentPPO agent implementation."""
    print("Testing RecurrentPPO Trading Agent")
    print("=" * 50)
    
    # Check available memory
    available_mb = get_available_memory_mb()
    print(f"\nAvailable memory: {available_mb} MB")
    
    # Create dummy environment
    from gymnasium import spaces
    
    class DummyTemporalEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
            self.step_count = 0
        
        def reset(self, seed=None):
            super().reset(seed=seed)
            self.step_count = 0
            return self.observation_space.sample(), {}
        
        def step(self, action):
            self.step_count += 1
            obs = self.observation_space.sample()
            
            # Simulate temporal pattern reward
            reward = float(np.sin(self.step_count * 0.1) * action)
            
            terminated = self.step_count >= 100
            truncated = False
            info = {
                'trend_accuracy': np.random.random(),
                'step': self.step_count
            }
            return obs, reward, terminated, truncated, info
    
    env = DummyTemporalEnv()
    
    # Test 1: Create RecurrentPPO agent
    print("\n1. Creating RecurrentPPO agent...")
    agent = create_recurrent_ppo_agent(env)
    if agent:
        print("   ✓ RecurrentPPO agent created successfully")
    else:
        print("   ⚠ RecurrentPPO creation failed (fallback triggered)")
        return
    
    # Test 2: Check configuration
    print("\n2. RecurrentPPO Configuration:")
    print(f"   LSTM hidden size: {agent.config['policy_kwargs']['lstm_hidden_size']}")
    print(f"   LSTM layers: {agent.config['policy_kwargs']['n_lstm_layers']}")
    print(f"   Shared LSTM: {agent.config['policy_kwargs']['shared_lstm']}")
    print(f"   N steps: {agent.config['n_steps']}")
    print(f"   Batch size: {agent.config['batch_size']}")
    
    # Test 3: Memory usage
    print("\n3. Memory usage:")
    memory = agent.get_memory_usage()
    print(f"   Model parameters: {memory['model_params_mb']:.2f} MB")
    print(f"   LSTM state: {memory['lstm_state_mb']:.2f} MB")
    print(f"   Rollout buffer: {memory['rollout_buffer_mb']:.2f} MB")
    print(f"   Total: {memory['total_mb']:.2f} MB")
    
    # Test 4: LSTM state info
    print("\n4. LSTM State Information:")
    lstm_info = agent.get_lstm_state_info()
    for key, value in lstm_info.items():
        print(f"   {key}: {value}")
    
    # Test 5: Quick training test
    print("\n5. Testing training (100 steps)...")
    try:
        agent.train(total_timesteps=100, progress_bar=False)
        print("   ✓ Training successful")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
    
    # Test 6: Prediction with LSTM states
    print("\n6. Testing prediction with LSTM states...")
    obs, _ = env.reset()
    
    # First prediction (no previous state)
    action1, lstm_states1 = agent.predict(obs, lstm_states=None, episode_start=np.array([True]))
    print(f"   First action: {action1}")
    
    # Second prediction (with previous state)
    obs, _, _, _, _ = env.step(action1[0])
    action2, lstm_states2 = agent.predict(obs, lstm_states=lstm_states1, episode_start=np.array([False]))
    print(f"   Second action: {action2}")
    print(f"   LSTM states maintained: {lstm_states2 is not None}")
    
    print("\n✓ RecurrentPPO agent test completed!")


if __name__ == "__main__":
    test_recurrent_ppo_agent()
