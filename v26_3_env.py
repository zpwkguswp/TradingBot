"""
V26.3.14 Master Fix - MTF Sniper Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- [V27 Phase 2] MTF Strategy: Added 1H Macro features (obs_dim=33)
- [V27 Phase 2] Macro Trend Filter: close vs 1H EMA-200 regimes
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from v26_0_env import V26_HeritageSniperEnv, HERITAGE_MDD_LIMIT
from v25_4_2_env import _tcn_entropy_ratio
from v25_env import _wilder_atr12
from v24_4_env import FEE_RATE

class V26_3_HeritageDisparityEnv(V26_HeritageSniperEnv):
    def __init__(
        self,
        data_dir: str,
        coin_files: list,
        split_type: str | None = None,
        obs_dim: int = 33, # [V27 Phase 2] 29 + 4 (h1_ema_20, h1_60, h1_200, h1_adx)
        adx_thr: float = 12.0,
        tcn_gate_thr: float = 0.55,
        hold_signal_thr: float = 0.05,
        consecutive_need: int = 1,
        leverage: int = 10,
        sl_atr_coef: float = 3.2,
        tp_atr_coef: float = 12.0,
        be_atr_coef: float = 1.5,
        close_th: float = 0.005,
        far_th: float = 2.5,
        target_profit: float = 0.008,
    ):
        super().__init__(
            data_dir,
            coin_files,
            split_type=split_type,
            obs_dim=obs_dim,
            adx_thr=adx_thr,
            tcn_gate_thr=tcn_gate_thr,
            hold_signal_thr=hold_signal_thr,
            consecutive_need=consecutive_need,
            leverage=leverage,
            sl_atr_coef=sl_atr_coef,
            tp_atr_coef=tp_atr_coef,
            be_atr_coef=be_atr_coef,
        )
        self.close_th = close_th
        self.far_th = far_th
        # [V27 Phase 2.6] Raised target_profit from 0.5% to 0.8%
        # 0.8% target: ROE 8% -> net_pnl ~6.2% vs SL loss ~5-7% => R:R ~1:1 (breakeven viable)
        self.target_profit = 0.008

        # ROOT FIX: Disable parent gates
        self.adx_thr      = 0.0   
        self.atr_raw_thr  = 0.0   
        self.tcn_gate_thr = 0.0   

        self._ensure_ema_indicators()
        
        # [V27 Phase 2.10] Lifetime Tracker (SB3 Auto-Reset Defense)
        # These variables persist through reset() to track multi-episode progress.
        self.lifetime_trades     = 0
        self.lifetime_goal_hits  = 0

    def _ensure_ema_indicators(self):
        for coin in self.coin_files:
            df = self.cached_dfs[coin]
            if 'ema_20' not in df.columns:
                df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            if 'ema_60' not in df.columns:
                df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
            df['v26_disparity'] = ((df['close'] - df['ema_60']) / df['ema_60']) * 100.0

    def _close_position(self) -> tuple:
        """[V27 Phase 2.3] Fix Reward Trap: Override parent to use pure net_pnl as reward."""
        _, won, mscl_penalty, net_pnl = super()._close_position()
        
        # Reward is simply the net profit (already includes fees)
        reward = net_pnl
        
        # [V27 Phase 2.9] Goal Success Bonus (+0.05)
        # Provides a bright target to overcome the Time Penalty friction.
        if net_pnl >= self.target_profit:
            reward += 0.05
            self.lifetime_goal_hits += 1

        self.lifetime_trades += 1

        if won:
            self.goal_hits += 1
            
        return float(reward), bool(won), mscl_penalty, float(net_pnl)

    def _get_obs(self) -> np.ndarray:
        """[V27 Phase 2] Append 1H Macro features."""
        base_obs = super()._get_obs() # (29,)
        row = self.df.iloc[self.current_step]
        
        # New MTF Features (4)
        h1_features = np.array([
            float(row.get("h1_ema_20", 0.0)) / float(row["close"]),
            float(row.get("h1_ema_60", 0.0)) / float(row["close"]),
            float(row.get("h1_ema_200", 0.0)) / float(row["close"]),
            float(row.get("h1_adx_14", 0.0))
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, h1_features])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.total_trades      = 0
        self.position_duration = 0
        self.goal_hits         = 0
        self.consecutive_count = 3
        self.last_signal_dir   = 0
        self.mscl_halt       = False
        self.mscl_halt_timer = 0
        return obs, info

    def step(self, action):
        if self.current_step >= len(self.df):
            return self.reset()[0], 0.0, True, False, {}

        idx       = self.df.index[self.current_step]
        row       = self.df.iloc[self.current_step]
        close_now = float(row["close"])
        disparity = float(row["v26_disparity"])

        # 1. Action Clipping
        action_val = float(np.clip(action[0], -1.0, 1.0))
        signal_dir = 0
        if action_val > self.hold_signal_thr:
            signal_dir = 1
        elif action_val < -self.hold_signal_thr:
            signal_dir = -1

        # 2. Minimal Masking: Extreme Exhaustion
        allow_long  = True
        allow_short = True
        if disparity > self.far_th:
            allow_long  = False
        if disparity < -self.far_th:
            allow_short = False

        if signal_dir == 1 and not allow_long:
            signal_dir = 0
        elif signal_dir == -1 and not allow_short:
            signal_dir = 0

        # 3. Macro Trend Filter (1H EMA-200)
        # Simplified rollback to prevent lagging filters (Short-term EMA) from causing top-buying.
        h1_ema200 = float(row.get("h1_ema_200", close_now))
        reward_penalty = 0.0
        
        entry_allowed = True
        if signal_dir == 1:
            if close_now < h1_ema200:
                entry_allowed = False
        elif signal_dir == -1:
            if close_now > h1_ema200:
                entry_allowed = False
                
        if not entry_allowed and signal_dir != 0:
            signal_dir = 0
            # [V27 Phase 2.9] Reduced entry rejection friction
            reward_penalty = -0.00005
            
        # [V27 Phase 2.6] Min Hold Duration shortened: 4 -> 2 steps (30 min)
        # 4-step hold was trapping bad trades for 1 hour, worsening loss size.
        # 2-step hold still filters pure noise while allowing faster exits.
        if self.position is not None and self.position_duration < 2:
            if signal_dir == 0:
                signal_dir = 1 if self.position["dir"] == "long" else -1

        # 4. Dynamic Leverage (Constrained 1.0 ~ 10.0)
        p_up          = float(row.get("v14_p_long", 1 / 3))
        entropy_ratio = 1.0 - abs(p_up - 0.5) * 2.0
        self.leverage = float(np.clip(10.0 * (1.0 - entropy_ratio), 1.0, 10.0))

        # 5. Execute Parent Step
        self.consecutive_count = 3    
        self.mscl_halt         = False
        self.mscl_halt_timer   = 0
        
        was_in_pos = self.position is not None
        
        # Convert signal_dir to action array for super()
        mock_action = np.array([float(signal_dir)], dtype=np.float32)
        obs, reward, done, truncated, info = super().step(mock_action)
        
        reward += reward_penalty
        is_in_pos  = self.position is not None

        # 6. Trade / Duration Tracking & Time Penalty
        if is_in_pos:
            # [V27 Phase 2.8] Time Penalty (-0.0001 per step)
            # Incentivizes hitting target_profit (0.8%) quickly or performing manual 'Soft Stop'.
            reward -= 0.0001
            
            if not was_in_pos or self.position.get("duration", 0) == 0:
                self.total_trades      += 1
                self.position_duration  = 0
                self.position["v26_peak_high"] = close_now
                self.position["v26_peak_low"]  = close_now
                reward -= 0.0002 # [V27 Phase 2.1] Reduced entry friction

                # [V27 Phase 2.11] Pullback Entry Reward / Chasing Penalty
                # price vs 1H EMA-200 disparity check
                entry_disparity = abs(close_now - h1_ema200) / (h1_ema200 + 1e-9)
                if entry_disparity <= 0.005:   # 0.5% (눌림목 타점)
                    reward += 0.02
                elif entry_disparity >= 0.01: # 1.0% (추격 매수 억제)
                    reward -= 0.01
            else:
                self.position_duration += 1
                self.position["v26_peak_high"] = max(self.position.get("v26_peak_high", close_now), close_now)
                self.position["v26_peak_low"] = min(self.position.get("v26_peak_low", close_now), close_now)
        else:
            self.position_duration = 0

        # 7. Exit Logic
        if is_in_pos and not done:
            side    = self.position["dir"]
            entry_p = self.position["entry_price"]
            raw_pnl_pct = (close_now / entry_p - 1.0) if side == "long" else (1.0 - close_now / entry_p)

            if raw_pnl_pct >= self.target_profit:
                atr_now = _wilder_atr12(self.df, self.current_step)
                drawback_limit = 0.7 * atr_now
                triggered = (
                    (side == "long"  and self.position["v26_peak_high"] - close_now >= drawback_limit) or
                    (side == "short" and close_now - self.position["v26_peak_low"]   >= drawback_limit)
                )
                if triggered:
                    # Our overridden _close_position already rewards net_pnl and increments goal_hits
                    r, w, p, n = self._close_position()
                    reward += r
                    self.position_duration = 0
            elif self.position_duration >= 96 and raw_pnl_pct < self.target_profit:
                r, w, p, n = self._close_position()
                reward += r
                self.position_duration = 0

        # 8. Holding Penalty
        if self.position:
            duration         = self.position["duration"]
            funding_friction = 0.0001 * duration * (self.leverage / 10.0)
            reward -= funding_friction

        info["dynamic_leverage"]     = self.leverage
        info["disparity"]            = disparity
        info["total_trades"]         = self.total_trades
        info["winning_trades"]       = self.winning_trades
        info["goal_hits"]            = self.goal_hits
        info["position_duration"]    = self.position_duration
        # [V27 Phase 2.10] IPC Bridge for SubprocVecEnv
        info["balance"]              = float(self.balance)
        info["lifetime_trades"]      = self.lifetime_trades
        info["lifetime_goal_hits"]   = self.lifetime_goal_hits

        return obs, reward, done, truncated, info
