"""
V26.0 Heritage-Sniper Environment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- V14 & V25.2 Hybrid Strategy
- High-Conviction PnL Maximization
- Triple Barrier Upgrade & 2.5x Fee Weight
- MDD Allowance: (0.7 * Return) - (0.3 * MDD_Penalty)
"""

from __future__ import annotations
import math
import numpy as np
from v25_4_2_env import (
    V25_4_2_BTC_Survival15mEnv,
    _wilder_atr12,
    _tcn_entropy_ratio,
    _utc_minute_from_row,
)
from v24_4_env import (
    CANDLES_PER_DAY,
    DAILY_TRADE_CAP,
    FEE_RATE,
    INITIAL_BALANCE,
    ROLLING_VOL_WIN,
    REWARD_MDD_COEF,
)

# V26.0 설정
HERITAGE_LEVERAGE_FIXED = 10  # 공격적 진입을 위한 10배 레버리지
HERITAGE_FEE_WEIGHT     = 2.5 # 수수료 가중치 정책 유지
HERITAGE_MDD_LIMIT      = 0.35 # MDD 소폭 상향 허용 (V25는 0.30)

class V26_HeritageSniperEnv(V25_4_2_BTC_Survival15mEnv):
    def __init__(
        self,
        data_dir: str,
        coin_files: list,
        split_type: str | None = None,
        obs_dim: int = 29,
        adx_thr: float = 12.0,
        tcn_gate_thr: float = 0.55, # V14 DNA: 높은 문턱값
        hold_signal_thr: float = 0.05,
        consecutive_need: int = 1,
        leverage: int = HERITAGE_LEVERAGE_FIXED,
        sl_atr_coef: float = 2.5,  # V14 DNA: 타이트한 손절
        tp_atr_coef: float = 12.0, # V14 DNA: 큰 익절 타겟
        be_atr_coef: float = 1.5,
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
            leverage=HERITAGE_LEVERAGE_FIXED,
            sl_atr_coef=sl_atr_coef,
            tp_atr_coef=tp_atr_coef,
            be_atr_coef=be_atr_coef,
        )
        self.mdd_penalty_coef = 0.3 # MDD 페널티 비중 30%
        self.return_coef = 0.7      # 수익 비중 70%

    def _close_position(self) -> tuple:
        if not self.position:
            return 0.0, False, 0.0, 0.0
            
        pnl_raw = float(self._get_pnl_raw())
        lev = float(self.position.get("leverage", self.leverage))
        fee_mult = min(lev, 10.0)
        
        # 기본 수수료 + 슬리피지 페널티
        fee = float(self.position["size"]) * FEE_RATE * fee_mult
        tx_cost = fee * 3.0 # 슬리피지 포함 3배 비용 적용
        net_pnl = pnl_raw - tx_cost
        
        self.balance *= 1.0 + net_pnl
        self.balance = max(self.balance, 0.0)
        
        won = net_pnl > 0
        self.total_trades += 1
        if won:
            self.winning_trades += 1
        
        mscl_penalty = float(self._compute_mscl_penalty(won))
        
        # [V26.0 Heritage Reward]
        # 순수익에 집중하되, 수수료 가중치를 직접 반영하여 "Perfect Entry" 유도
        net_return_pct = net_pnl / (self.position["size"] * lev + 1e-9)
        
        # 수익 구간 보너스 (V14 DNA)
        reward = net_pnl * self.return_coef
        if net_return_pct > 0.02: # 2% 이상 순수익 시 추가 보너스
            reward += 0.1
            
        # 수수료 기반 페널티
        reward -= tx_cost * HERITAGE_FEE_WEIGHT

        self.position = None
        self.running_max_equity = 0.0
        return float(reward), bool(won), mscl_penalty, float(net_pnl)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # MDD 및 최종 보상 재조정
        if not done:
            new_equity = self._get_equity()
            self.peak_equity = max(self.peak_equity, new_equity)
            current_dd = (self.peak_equity - new_equity) / (self.peak_equity + 1e-9)
            
            # MDD 페널티 (0.3 가중치)
            reward -= self.mdd_penalty_coef * current_dd
            
            # MDD 리밋 체크 (HERITAGE_MDD_LIMIT)
            if current_dd > HERITAGE_MDD_LIMIT:
                done = True
                return obs, -500.0, done, truncated, {**info, "mdd_breach": True}
                
        return obs, reward, done, truncated, info
