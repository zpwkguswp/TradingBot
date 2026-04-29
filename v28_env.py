"""
V28 Phase 2.1: MFE-Driven Learning Environment (36-Dim Observation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- [MFE/MAE Tracking] Tracks max favorable/adverse excursion during position.
- [Regret Error Penalty] Penalizes (MFE - Actual_PnL) on close.
- [MTF Optimization] Designed for multi-timeframe optimization via v28_train.py.
- [Phase 2.1] Observation expanded 33 -> 36: +macro_adx, +macro_atr_ratio, +macro_bb_width
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from gymnasium import spaces
from v26_3_env import V26_3_HeritageDisparityEnv

# [Phase 2.1] 36차원 관측 공간 상수
_OBS_DIM = 36
_MACRO_COLS = ("macro_adx", "macro_atr_ratio", "macro_bb_width")

class V28_MFE_Env(V26_3_HeritageDisparityEnv):
    def __init__(self, *args, **kwargs):
        # 🌟 [신규] 3대 암세포 억제제 파라미터 수신부
        self.adx_th = kwargs.pop("adx_th", 0.20)
        self.max_disp = kwargs.pop("max_disp", 0.05)
        self.trail_act = kwargs.pop("trail_act", 0.03)

        # 부모에게 obs_dim=33으로 초기화 후 observation_space를 36으로 덮어씀
        super().__init__(*args, **kwargs)

        # [Phase 2.1] observation_space 를 36차원으로 확장
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )

        self.mfe = 0.0
        self.mae = 0.0
        self.entry_price = 0.0
        self.side = None
        self.exit_reason = "None"
        self.entry_disparity = 0.0

        # [Telemetry] 진입 순간 스냅샷 변수
        self.snapshot_adx = np.nan
        self.snapshot_bb_width = np.nan

        # [Phase 2.1] MDD / Equity 추적 변수
        self._initial_balance_ref = None  # reset() 첫 호출 시 확정
        self._peak_balance = None
        self._max_mdd = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.mfe = 0.0
        self.mae = 0.0
        self.entry_price = 0.0
        self.side = None
        self.exit_reason = "None"
        self.entry_disparity = 0.0
        self.snapshot_adx = np.nan
        self.snapshot_bb_width = np.nan

        # MDD 추적 초기화: 에피소드 시작 시 잔고를 기준으로 설정
        cur_bal = float(getattr(self, 'balance', 10_000.0))
        self._initial_balance_ref = cur_bal
        self._peak_balance = cur_bal
        self._max_mdd = 0.0
        return obs, info

    def step(self, action):
        # 🌟 [신규] 사령관의 강제 통제 (진입 결재 단계)
        # 횡보장(진흙탕)이거나 너무 높은 고점(정수리)이면 액션을 중립(0)으로 뺏음
        if getattr(self, "position", None) is None:
            action_val = action[0] if isinstance(action, np.ndarray) else action
            action_val = float(np.clip(action_val, -1.0, 1.0))
            if abs(action_val) > getattr(self, "hold_signal_thr", 0.05):
                if self.current_step < len(self.df):
                    row = self.df.iloc[self.current_step]
                    cur_adx = float(row.get("macro_adx", 1.0))
                    h1_ema200 = float(row.get("h1_ema_200", row["close"]))
                    cur_disp = abs(row["close"] - h1_ema200) / (h1_ema200 + 1e-9)
                    
                    if cur_adx < self.adx_th or cur_disp > self.max_disp:
                        if isinstance(action, np.ndarray):
                            action = np.zeros_like(action)
                        else:
                            action = 0.0

        obs, reward, done, truncated, info = super().step(action)
        
        # MFE / MAE Tracking
        if self.position is not None:
            close_now = float(self.df.iloc[self.current_step]["close"])
            if self.entry_price == 0.0:
                self.entry_price = self.position["entry_price"]
                self.side = self.position["dir"]
                self.mfe = 0.0
                self.mae = 0.0
                
                # Calculate Entry Disparity (vs Macro EMA-200)
                row = self.df.iloc[self.current_step]
                h1_ema200 = float(row.get("h1_ema_200", self.entry_price))
                self.entry_disparity = abs(self.entry_price - h1_ema200) / (h1_ema200 + 1e-9)

                # [Telemetry] 진입 순간의 거시 지표 그늙로 박제
                self.snapshot_adx      = float(row.get("macro_adx",      np.nan))
                self.snapshot_bb_width = float(row.get("macro_bb_width",  np.nan))
            
            # Calculate current PnL %
            current_pnl_raw = (close_now / self.entry_price - 1.0) if self.side == "long" else (1.0 - close_now / self.entry_price)
            
            # Update MFE / MAE
            self.mfe = max(self.mfe, current_pnl_raw)
            self.mae = min(self.mae, current_pnl_raw)

            # 🌟 [신규] 스마트 트레일링 방어 체계 (익절 도우미)
            if not done:
                if self.mfe >= self.trail_act:
                    preservation_line = self.mfe * 0.7
                    if current_pnl_raw < preservation_line:
                        # 멱살 잡고 익절!
                        r, w, m, n = self._close_position()
                        reward += r
                        if hasattr(self, "last_trade_metrics"):
                            self.last_trade_metrics["Exit_Reason"] = "Trailing_Stop"
        
        # Check if a trade was just closed and push metrics to info
        last_trade = self.get_last_trade_info()
        if last_trade:
            for k, v in last_trade.items():
                info[f"trade_{k}"] = v
            info["trade_closed"] = True

        # ── MDD / Equity / Trades 주입 ────────────────────────────────────
        cur_bal = float(getattr(self, 'balance', 10_000.0))

        # 첫 step에서 기준 잔고가 아직 없으면 초기화
        if self._peak_balance is None:
            self._peak_balance = cur_bal
            self._initial_balance_ref = cur_bal

        # 고점 갱신
        if cur_bal > self._peak_balance:
            self._peak_balance = cur_bal

        # 현재 낙폭 및 MDD 갱신
        drawdown = (self._peak_balance - cur_bal) / (self._peak_balance + 1e-9)
        if drawdown > self._max_mdd:
            self._max_mdd = drawdown

        init_bal = self._initial_balance_ref if self._initial_balance_ref else cur_bal
        info["bot_equity"]   = cur_bal / (init_bal + 1e-9)   # 1.0 = 원금 유지
        info["bot_mdd"]      = float(self._max_mdd)           # 0.15 = 15% MDD
        info["total_trades"] = int(getattr(self, 'total_trades', 0))
        # ─────────────────────────────────────────────────────────────────

        return obs, reward, done, truncated, info

    def _get_obs(self) -> np.ndarray:
        """[Phase 2.1] 부모 33차원 obs에 macro 3개 특성을 추가하여 36차원 반환."""
        base_obs = super()._get_obs()  # (33,)
        row = self.df.iloc[self.current_step]

        # macro 컬럼이 존재하면 실값 사용, 없으면 0.0 패딩
        if "macro_adx" in self.df.columns:
            macro = np.array([
                float(row["macro_adx"]),
                float(row["macro_atr_ratio"]),
                float(row["macro_bb_width"]),
            ], dtype=np.float32)
        else:
            macro = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([base_obs, macro])  # (36,)
        return obs.astype(np.float32)

    def _close_position(self) -> tuple:
        """Override to implement Regret Error penalty and Precision Logging."""
        # 1. Call parent to get basis results
        reward, won, mscl_penalty, net_pnl = super()._close_position()
        
        # 2. Update MFE to resolve the Paradox (MFE must be >= Actual PnL)
        # We ensure mfe is at least the net_pnl achieved at the moment of exit.
        self.mfe = max(self.mfe, net_pnl)
        
        # 3. Calculate Regret Error (always positive penalty)
        regret = max(0.0, self.mfe - net_pnl)
        regret_penalty = regret * 0.5
        reward -= regret_penalty
        
        # 4. Precision Logging Info
        # Clip capture_ratio to [0, 1] to ensure mathematical consistency.
        if self.mfe > 0:
            capture_ratio = np.clip(net_pnl / self.mfe, 0.0, 1.0)
        elif self.mfe < 0 and net_pnl < 0:
            # If both are negative (loss), a higher MFE (less negative) means better capture of the peak.
            # But usually Capture Ratio is defined for profitable trades or positive MFE.
            # For simplicity and per user request, we clip it.
            capture_ratio = 0.0
        else:
            capture_ratio = 0.0
        
        # Determine Exit Reason
        exit_reason = "Soft"
        if net_pnl <= -0.01: 
            exit_reason = "SL"
        elif net_pnl >= self.target_profit:
            exit_reason = "TP"

        self.last_trade_metrics = {
            "MFE": float(self.mfe),
            "MAE": float(self.mae),
            "Actual_PnL": float(net_pnl),
            "Capture_Ratio": float(capture_ratio),
            "Exit_Reason": exit_reason,
            "Holding_Steps": int(self.position_duration),
            "Entry_Price": float(self.entry_price),
            "Exit_Price": float(self.df.iloc[self.current_step]["close"]),
            "Entry_Disparity": float(self.entry_disparity),
            # [Telemetry] 진입 스냅샷
            "entry_adx":      self.snapshot_adx,
            "entry_bb_width": self.snapshot_bb_width,
        }
        
        # Store state for logging before reset
        metrics_to_return = (float(reward), bool(won), mscl_penalty, float(net_pnl))

        # Reset trackers for next trade
        self.mfe = 0.0
        self.mae = 0.0
        self.entry_price = 0.0
        self.side = None
        
        return metrics_to_return

    def get_last_trade_info(self):
        if hasattr(self, "last_trade_metrics"):
            metrics = self.last_trade_metrics
            # Clean up after providing
            delattr(self, "last_trade_metrics")
            return metrics
        return None
