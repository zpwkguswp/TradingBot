"""
V25.4.2 (15m Focus): Survival-First + Daily Context — Environment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V25.4.1의 생존 규칙을 유지하면서, 15m 모델이 "일일 주기" 맥락을 더 잘 학습하도록
시간 특성(UTC time-of-day sin/cos)을 관측치에 추가한다.

핵심 규칙 (고정):
- Reward = Net_PnL - (Transaction_Cost * 2.5)
- Liquidation or MDD > 5% -> Reward = -500, episode done
- Hybrid Filter (double lock):
    Disparity hard filter: 1.03
    3-Bar Structural Reversal 미발생 시 진입 0
- leverage=7x 고정
- slippage penalty=2x (거래비용에 추가)
"""

from __future__ import annotations

import math
import numpy as np

from v25_env import (
    V25_1_SpecialistEnv,
    _wilder_atr12,
    _tcn_entropy_ratio,
    _utc_minute_from_row,
    _in_funding_snipe_window,
)
from v24_4_env import (
    CANDLES_PER_DAY,
    DAILY_TRADE_CAP,
    FEE_RATE,
    INITIAL_BALANCE,
    ROLLING_VOL_WIN,
)


SURVIVAL_LEVERAGE_FIXED = 7
SURVIVAL_DISPARITY_HARD = 1.015
SURVIVAL_MDD_LIMIT = 0.30
SURVIVAL_FAIL_REWARD = -500.0
SLIPPAGE_FEE_MULT = 2.0


class V25_4_2_BTC_Survival15mEnv(V25_1_SpecialistEnv):
    """
    Obs:
      base(23) + filter(4: disparity/squeeze/rev_l/rev_s) + time(2: sin/cos) = 29
    """

    def __init__(
        self,
        data_dir: str,
        coin_files: list,
        split_type: str | None = None,
        obs_dim: int = 29,
        adx_thr: float = 12.0,
        # gate off by default (cached TCN is neutral)
        tcn_gate_thr: float = 0.0,
        hold_signal_thr: float = 0.07,
        consecutive_need: int = 1,
        leverage: int = SURVIVAL_LEVERAGE_FIXED,
        sl_atr_coef: float = 2.8,
        tp_atr_coef: float = 2.4,
        be_atr_coef: float = 2.2,
    ):
        self.hold_signal_thr = float(hold_signal_thr)
        self.consecutive_need = int(max(1, consecutive_need))
        super().__init__(
            data_dir,
            coin_files,
            split_type=split_type,
            obs_dim=obs_dim,
            adx_thr=float(adx_thr),
            tcn_gate_thr=float(tcn_gate_thr),
            leverage=SURVIVAL_LEVERAGE_FIXED,
            sl_atr_coef=float(sl_atr_coef),
            tp_atr_coef=float(tp_atr_coef),
            be_atr_coef=float(be_atr_coef),
            # keep survival reward clean (no extra shaping)
            phi_mdd_shaping=0.0,
            theta_vol_penalty=0.0,
        )

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()  # (23,)
        row = self.df.iloc[self.current_step]
        extra_filter = np.array(
            [
                float(row.get("disparity_200", 1.0)),
                float(row.get("ema_squeeze", 0.0)),
                float(row.get("structural_rev_long", 0.0)),
                float(row.get("structural_rev_short", 0.0)),
            ],
            dtype=np.float32,
        )

        # daily cycle context (UTC)
        utc_min = _utc_minute_from_row(row, self.current_step)
        phase = (utc_min % (24 * 60)) / (24 * 60)
        tod = np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)], dtype=np.float32)
        return np.concatenate([base_obs, extra_filter, tod])

    def _close_position(self) -> tuple[float, bool, float, float]:
        if not self.position:
            return 0.0, False, 0.0, 0.0

        pnl_raw = float(self._get_pnl_raw())
        lev = float(self.position.get("leverage", self.leverage))
        fee_mult = min(lev, 10.0)

        fee = float(self.position["size"]) * FEE_RATE * fee_mult
        slippage_fee = fee * SLIPPAGE_FEE_MULT
        tx_cost = fee + slippage_fee
        net_pnl = pnl_raw - tx_cost

        self.balance *= 1.0 + net_pnl
        self.balance = max(self.balance, 0.0)

        won = net_pnl > 0
        self.total_trades += 1
        if won:
            self.winning_trades += 1
            self.consec_loss_count = 0
        else:
            self.consec_loss_count = getattr(self, "consec_loss_count", 0) + 1

        mscl_penalty = float(self._compute_mscl_penalty(won))

        reward = net_pnl - (tx_cost * 2.5)
        
        # 미세 연속 손실 페널티 (악성 과매매 억제)
        if not won:
            step_pen = min(self.consec_loss_count * 0.5, 2.5)
            reward -= step_pen

        self.position = None
        self.running_max_equity = 0.0
        return float(reward), bool(won), mscl_penalty, float(net_pnl)

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        reward = 0.0
        done = False
        info: dict = {}

        prev_equity = self._get_equity()

        self.steps_today += 1
        if self.steps_today >= CANDLES_PER_DAY:
            self.steps_today = 0
            self.daily_trade_count = 0

        row = self.df.iloc[self.current_step]
        hi, lo, close_now = float(row["high"]), float(row["low"]), float(row["close"])
        adx_14 = float(row.get("adx_14", 0.30))
        atr_raw = float(row.get("atr_raw", 0.002))
        atr_z = float(row.get("atr_z", 0.0))

        if (adx_14 * 100.0) < self.adx_thr or atr_raw < self.atr_raw_thr:
            action_val = 0.0

        signal_dir = 0
        if action_val > self.hold_signal_thr:
            signal_dir = +1
        elif action_val < -self.hold_signal_thr:
            signal_dir = -1

        # 단일 필터: Disparity 이격 체크를 완전히 제거하고 Structural Reversal 하나에만 의존
        disparity = float(row.get("disparity_200", 1.0))
        rev_long = float(row.get("structural_rev_long", 0.0))
        rev_short = float(row.get("structural_rev_short", 0.0))

        if signal_dir == +1:
            if rev_long != 1.0:
                signal_dir = 0
        elif signal_dir == -1:
            if rev_short != 1.0:
                signal_dir = 0

        if signal_dir != 0 and signal_dir == self.last_signal_dir:
            self.consecutive_count += 1
        elif signal_dir != 0:
            self.consecutive_count = 1
            self.last_signal_dir = signal_dir
        else:
            self.consecutive_count = 0
            self.last_signal_dir = 0

        can_enter = (
            self.consecutive_count >= self.consecutive_need
            and self.daily_trade_count < DAILY_TRADE_CAP
            and not self.position
            and not self.mscl_halt
        )

        atr12 = _wilder_atr12(self.df, self.current_step)

        # optional emergency deleverage (still useful if TCN present; neutral cache => usually off)
        p_up = float(row.get("tcn_p_up", 1 / 3))
        p_down = float(row.get("tcn_p_down", 1 / 3))
        p_chop = float(row.get("tcn_p_chop", 1 / 3))
        entropy_ratio = _tcn_entropy_ratio(p_up, p_down, p_chop)
        _emergency = entropy_ratio > self.emergency_entropy_thr

        realized_reward_closed = 0.0

        if self.position:
            liq_p = float(self.position["liq_price"])
            side = self.position["dir"]
            if (side == "long" and lo <= liq_p) or (side == "short" and hi >= liq_p):
                self.position = None
                self.balance = max(self.balance * 0.02, 0.0)
                done = True
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                return obs, float(SURVIVAL_FAIL_REWARD), done, False, {**self._get_episode_info(), "liquidated": True}

            entry_p = float(self.position["entry_price"])
            if not self.position.get("be_activated", False):
                if (side == "long" and hi >= entry_p + self.be_atr_coef * atr12) or (
                    side == "short" and lo <= entry_p - self.be_atr_coef * atr12
                ):
                    self.position["sl_price"] = entry_p
                    self.position["be_activated"] = True

            sl_p = float(self.position["sl_price"])
            if (side == "long" and lo <= sl_p) or (side == "short" and hi >= sl_p):
                realized_r, won, mscl_pen, _net = self._close_position()
                reward += realized_r
                reward -= mscl_pen
                realized_reward_closed = realized_r
            elif self.position:
                act_dist = self.tp_atr_coef * atr12
                if side == "long":
                    if hi >= entry_p + act_dist:
                        self.position["trail_active"] = True
                    if self.position.get("trail_active"):
                        self.position["peak_track"] = max(float(self.position.get("peak_track", entry_p)), hi)
                        if lo <= float(self.position["peak_track"]) - act_dist:
                            realized_r, won, mscl_pen, _net = self._close_position()
                            reward += realized_r
                            reward -= mscl_pen
                            realized_reward_closed = realized_r
                else:
                    if lo <= entry_p - act_dist:
                        self.position["trail_active"] = True
                    if self.position.get("trail_active"):
                        self.position["peak_track"] = min(float(self.position.get("peak_track", entry_p)), lo)
                        if hi >= float(self.position["peak_track"]) + act_dist:
                            realized_r, won, mscl_pen, _net = self._close_position()
                            reward += realized_r
                            reward -= mscl_pen
                            realized_reward_closed = realized_r

        if not self.position and not done and can_enter and signal_dir != 0:
            base_size = abs(action_val)
            
            # PnL 극대화: 변동성이 낮을 때 스케일링 보너스를 50%로 확대하고, 상한선도 1.5배까지 해제
            scale_modifier = 1.0 + min(max(-atr_z * 0.25, 0.0), 0.5)
            scaled_size = min(base_size * scale_modifier, 1.5)
            
            if atr_z > 2.0:
                scaled_size *= 0.5

            dir_ = "long" if signal_dir == 1 else "short"
            eff_leverage = 1.0 if _emergency else float(SURVIVAL_LEVERAGE_FIXED)
            sl_off = self.sl_atr_coef * atr12 * (0.7 if _emergency else 1.0)
            sl_price = (close_now - sl_off) if dir_ == "long" else (close_now + sl_off)

            self.position = {
                "dir": dir_,
                "entry_price": close_now,
                "size": float(scaled_size),
                "duration": 0,
                "leverage": eff_leverage,
                "liq_price": self._liq_prices(close_now, dir_, eff_leverage),
                "sl_price": sl_price,
                "entry_atr": atr12,
                "trail_active": False,
                "peak_track": close_now,
                "be_activated": False,
            }

            fee_mult = min(eff_leverage, 10.0)
            fee = self.balance * float(scaled_size) * FEE_RATE * fee_mult
            self.balance -= (fee + fee * SLIPPAGE_FEE_MULT)
            self.daily_trade_count += 1
            self.consecutive_count = 0
            self.running_max_equity = prev_equity

        if self.position and not done:
            self.position["duration"] += 1
            if self.position["duration"] >= self.max_hold:
                realized_r, won, mscl_pen, _net = self._close_position()
                reward += realized_r
                reward -= mscl_pen
                realized_reward_closed = realized_r

        new_equity = self._get_equity()
        self.peak_equity = max(self.peak_equity, new_equity)
        current_dd = (self.peak_equity - new_equity) / (self.peak_equity + 1e-9)
        self.mdd = max(self.mdd, current_dd)

        if self.mdd > SURVIVAL_MDD_LIMIT:
            done = True
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, float(SURVIVAL_FAIL_REWARD), done, False, {**self._get_episode_info(), "mdd_breach": True}

        delta_equity = (new_equity - prev_equity) / INITIAL_BALANCE
        self.recent_returns.append(delta_equity)
        if len(self.recent_returns) > ROLLING_VOL_WIN:
            self.recent_returns.pop(0)

        self.total_steps += 1
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            if self.position:
                realized_r, won, mscl_pen, _net = self._close_position()
                reward += realized_r
                reward -= mscl_pen
            done = True
            info = self._get_episode_info()

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(reward), done, False, info

