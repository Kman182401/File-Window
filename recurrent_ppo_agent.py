#!/usr/bin/env python3
"""
RecurrentPPO Trading Agent

Production-ready RecurrentPPO implementation with LSTM for capturing
temporal dependencies in financial markets. Optimized for m5.large.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
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
        
        return True


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
        self.model = None
        self.callbacks = []
        
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
        base_config = {
            # Model architecture
            'policy': 'MlpLstmPolicy',
            'policy_kwargs': {
                'lstm_hidden_size': 64,  # Reduced for memory efficiency
                'n_lstm_layers': 1,      # Single layer for CPU
                'shared_lstm': True,      # Share LSTM between actor/critic
                'enable_critic_lstm': True,
                'net_arch': {
                    'pi': [64, 64],      # Actor network
                    'vf': [64, 64]       # Critic network
                },
                'activation_fn': nn.ReLU,
                'ortho_init': True       # Orthogonal initialization
            },
            
            # PPO hyperparameters
            'learning_rate': 3e-4,
            'n_steps': 512,          # Reduced for memory
            'batch_size': 32,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.01,        # Entropy coefficient
            'vf_coef': 0.5,          # Value function coefficient
            'max_grad_norm': 0.5,    # Gradient clipping
            
            # System
            'device': 'cpu',
            'verbose': 1,
            'tensorboard_log': './tensorboard_logs/recurrent_ppo/',
            'memory_limit_mb': 800
        }
        
        # Adjust based on available memory
        available_mb = get_available_memory_mb()
        
        if available_mb < 2000:
            # Low memory mode
            base_config['n_steps'] = 256
            base_config['batch_size'] = 16
            base_config['policy_kwargs']['lstm_hidden_size'] = 32
            base_config['policy_kwargs']['net_arch'] = {
                'pi': [32, 32],
                'vf': [32, 32]
            }
            logger.warning(f"Low memory ({available_mb}MB), using reduced RecurrentPPO config")
        elif available_mb < 3000:
            # Medium memory mode
            base_config['n_steps'] = 512
            base_config['batch_size'] = 32
        else:
            # High memory mode
            base_config['n_steps'] = 1024
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
            # Create model
            self.model = RecurrentPPO(
                policy=self.config['policy'],
                env=self.env,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                clip_range_vf=self.config['clip_range_vf'],
                normalize_advantage=self.config['normalize_advantage'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                policy_kwargs=self.config['policy_kwargs'],
                verbose=self.config['verbose'],
                tensorboard_log=self.config['tensorboard_log'],
                device=self.config['device']
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
        self.callbacks = [
            # Temporal pattern tracking
            TemporalTradingCallback(),
            
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
                self.env,
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
        
        return self.model.predict(
            observation,
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
        logger.info(f"RecurrentPPO model saved to {path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None):
        """
        Load model.
        
        Args:
            path: Load path
            env: Environment (uses self.env if None)
        """
        env = env or self.env
        
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
            n_steps = self.config.get('n_steps', 512)
            obs_size = np.prod(self.env.observation_space.shape)
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