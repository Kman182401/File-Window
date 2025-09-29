"""
SuperModel: Aggregates ML and RL signals into unified trading decisions.

This module provides a simple but extensible framework for combining
multiple model signals (ML predictions + RL actions) into a single
trading decision with clear rationale logging.
"""
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class SuperModel:
    """
    Aggregates ML and RL signals into unified trading decisions.
    
    Currently implements simple weighted voting, but designed for easy extension
    to more sophisticated ensemble methods.
    """
    
    def __init__(self, ml_weight: float = 0.6, rl_weight: float = 0.4):
        """
        Initialize SuperModel with configurable weights.
        
        Args:
            ml_weight: Weight for ML model predictions (0.0 to 1.0)
            rl_weight: Weight for RL model actions (0.0 to 1.0)
        """
        if abs(ml_weight + rl_weight - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: ML={ml_weight}, RL={rl_weight}")
        
        self.ml_weight = ml_weight
        self.rl_weight = rl_weight
        self.decisions_log = []
        
    def aggregate_signals(
        self, 
        ml_predictions: Dict[str, Any], 
        rl_actions: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        replay_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Combine ML and RL signals into a single trading decision.
        
        Args:
            ml_predictions: Dict with ML model predictions and confidence scores
            rl_actions: Dict with RL actions and value estimates
            market_context: Optional market conditions (volatility, regime, etc.)
            replay_mode: If True, indicates this is using historical replay data
            
        Returns:
            Dict containing decision, confidence, and rationale
        """
        # Extract primary signals
        ml_signal = self._extract_ml_signal(ml_predictions)
        rl_signal = self._extract_rl_signal(rl_actions)
        
        # Apply weighted voting
        weighted_score = (ml_signal * self.ml_weight + rl_signal * self.rl_weight)
        
        # Apply market context adjustments if in replay mode with real data
        if replay_mode and market_context:
            weighted_score = self._apply_market_context_adjustments(weighted_score, market_context)
        
        # Convert to trading decision
        if weighted_score > 0.5:
            decision = "BUY"
            confidence = weighted_score
        elif weighted_score < -0.5:
            decision = "SELL"
            confidence = abs(weighted_score)
        else:
            decision = "HOLD"
            confidence = 1.0 - abs(weighted_score)
            
        # Build decision rationale
        rationale = {
            "ml_signal": ml_signal,
            "rl_signal": rl_signal,
            "weighted_score": weighted_score,
            "ml_weight": self.ml_weight,
            "rl_weight": self.rl_weight,
            "ml_details": ml_predictions,
            "rl_details": rl_actions,
            "market_context": market_context or {},
            "replay_mode": replay_mode
        }
        
        result = {
            "decision": decision,
            "confidence": confidence,
            "rationale": rationale,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z")
        }
        
        # Log the decision
        self.decisions_log.append(result)
        logger.info(f"SuperModel Decision: {decision} (confidence: {confidence:.3f}, "
                   f"ML: {ml_signal:.3f}, RL: {rl_signal:.3f})")
        
        return result
    
    def _extract_ml_signal(self, ml_predictions: Dict[str, Any]) -> float:
        """Extract normalized signal from ML predictions."""
        if not ml_predictions:
            return 0.0
            
        # If we have multiple ML model predictions, average them
        if "ensemble_prediction" in ml_predictions:
            pred = ml_predictions["ensemble_prediction"]
        elif "prediction" in ml_predictions:
            pred = ml_predictions["prediction"]
        elif isinstance(ml_predictions, dict):
            # Take average of all numeric prediction values
            preds = []
            for key, val in ml_predictions.items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    preds.append(float(val))
            pred = np.mean(preds) if preds else 0.0
        else:
            pred = 0.0
            
        # Convert to signal range [-1, 1]
        if isinstance(pred, (int, np.integer)):
            # Discrete class predictions (0, 1, 2) -> (-1, 0, 1)
            if pred == 0:
                return -1.0  # SELL
            elif pred == 1:
                return 1.0   # BUY
            else:
                return 0.0   # HOLD
        else:
            # Continuous predictions - assume already in reasonable range
            return np.clip(float(pred), -1.0, 1.0)
    
    def _extract_rl_signal(self, rl_actions: Dict[str, Any]) -> float:
        """Extract normalized signal from RL actions."""
        if not rl_actions:
            return 0.0
            
        # Extract action from RL output
        if "action" in rl_actions:
            action = rl_actions["action"]
        elif "prediction" in rl_actions:
            action = rl_actions["prediction"]
        else:
            return 0.0
            
        # Convert discrete RL actions to signals
        # Assuming standard RL action space: 0=HOLD, 1=BUY, 2=SELL
        if action == 0:
            return 0.0   # HOLD
        elif action == 1:
            return 1.0   # BUY
        elif action == 2:
            return -1.0  # SELL
        else:
            # For continuous actions, clip to [-1, 1]
            return np.clip(float(action), -1.0, 1.0)
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return all logged decisions."""
        return self.decisions_log.copy()
    
    def clear_history(self):
        """Clear decision history."""
        self.decisions_log.clear()
    
    def save_decisions_to_jsonl(self, filepath: str):
        """Save decision history to JSONL file."""
        with open(filepath, 'w') as f:
            for decision in self.decisions_log:
                f.write(json.dumps(decision) + '\n')
                
    def _apply_market_context_adjustments(self, weighted_score: float, market_context: Dict[str, Any]) -> float:
        """Apply market context adjustments to weighted score when in replay mode"""
        try:
            adjusted_score = weighted_score
            
            # Volatility regime adjustment
            volatility_regime = market_context.get("volatility_regime", "normal")
            if volatility_regime == "high":
                # Reduce position sizing in high volatility
                adjusted_score *= 0.8
            elif volatility_regime == "low":
                # Can be slightly more aggressive in low volatility
                adjusted_score *= 1.1
            
            # Spread adjustment - avoid trading in wide spreads
            spread_bps = market_context.get("spread_bps", 1.0)
            if spread_bps > 3.0:  # Wide spread
                # Reduce signal strength due to high transaction costs
                spread_penalty = min(0.3, (spread_bps - 1.0) / 10.0)  # Max 30% penalty
                adjusted_score *= (1.0 - spread_penalty)
            
            # Market impact adjustment
            market_impact_bps = market_context.get("market_impact_bps", 0.1)
            if market_impact_bps > 1.0:  # High impact
                impact_penalty = min(0.2, (market_impact_bps - 0.1) / 5.0)  # Max 20% penalty
                adjusted_score *= (1.0 - impact_penalty)
            
            # Time-based adjustments (avoid trading at session boundaries, etc.)
            timestamp = market_context.get("timestamp")
            if timestamp and isinstance(timestamp, str):
                try:
                    ts = pd.to_datetime(timestamp)
                    hour = ts.hour
                    
                    # Reduce signal strength during typically low-liquidity hours
                    if hour < 9 or hour > 16:  # Outside regular trading hours
                        adjusted_score *= 0.9
                except:
                    pass  # Ignore timestamp parsing errors
            
            # Price momentum consideration
            if "open" in market_context and "close" in market_context:
                try:
                    price_change = (market_context["close"] - market_context["open"]) / market_context["open"]
                    
                    # If we have strong momentum in the signal direction, slightly boost
                    if (weighted_score > 0 and price_change > 0.001) or (weighted_score < 0 and price_change < -0.001):
                        adjusted_score *= 1.05  # Small momentum boost
                    elif (weighted_score > 0 and price_change < -0.001) or (weighted_score < 0 and price_change > 0.001):
                        adjusted_score *= 0.95  # Small momentum penalty
                except:
                    pass  # Ignore calculation errors
            
            # Ensure adjusted score stays within reasonable bounds
            adjusted_score = np.clip(adjusted_score, -2.0, 2.0)
            
            logger.debug(f"Market context adjustment: {weighted_score:.3f} -> {adjusted_score:.3f} "
                        f"(vol: {volatility_regime}, spread: {spread_bps:.1f}bps)")
            
            return adjusted_score
            
        except Exception as e:
            logger.error(f"Error applying market context adjustments: {e}")
            return weighted_score  # Return original score on error
                
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of decisions made."""
        if not self.decisions_log:
            return {"total_decisions": 0}
            
        decisions = [d["decision"] for d in self.decisions_log]
        confidences = [d["confidence"] for d in self.decisions_log]
        
        return {
            "total_decisions": len(decisions),
            "buy_count": decisions.count("BUY"),
            "sell_count": decisions.count("SELL"), 
            "hold_count": decisions.count("HOLD"),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }


def create_default_super_model() -> SuperModel:
    """Create SuperModel with default configuration."""
    return SuperModel(ml_weight=0.6, rl_weight=0.4)


def simulate_paper_trade(decision: Dict[str, Any], current_price: float, 
                        position_size: int = 1) -> Dict[str, Any]:
    """
    Simulate a paper trade based on SuperModel decision.
    
    Args:
        decision: SuperModel decision dict
        current_price: Current market price
        position_size: Number of contracts/shares
        
    Returns:
        Dict with trade simulation details
    """
    trade_sim = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "decision": decision["decision"],
        "confidence": decision["confidence"],
        "price": current_price,
        "size": position_size,
        "trade_type": "PAPER_SIMULATION",
        "rationale": decision["rationale"]
    }
    
    logger.info(f"PAPER TRADE: {decision['decision']} {position_size} @ {current_price:.2f} "
               f"(confidence: {decision['confidence']:.3f})")
    
    return trade_sim
