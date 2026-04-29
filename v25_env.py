"""
V25.1 Specialist — 하이브리드 방어 시스템 (Hybrid Defense System)
──────────────────────────────────────────────────────────────────
[신규] Break-even + Sliding Barrier:
  가격이 entry + ATR12*be_atr_coef 터치 → SL을 entry(본절)로 이동
  이후 고점 갱신마다 peak - ATR12*tp_atr_coef 로 Sliding Trail 강화

[신규] 사후 검증 보상 (r_sl):
  EXIT 후 5봉 추적 → Good Exit(방향 지속) = 보너스 / Bad Exit(반전) = 페널티

[신규] Risk Sensor (Emergency Deleveraging):
  TCN 출력 엔트로피 > 임계치 → 즉시 Leverage 1x 강제 + SL 30% 타이트 조정

Optuna 탐색 (6개):
  leverage, adx_thr, tcn_gate_thr, sl_atr_coef, tp_atr_coef, be_atr_coef
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from v24_4_env import (
    CANDLES_PER_DAY,
    CONSECUTIVE_NEED,
    DAILY_TRADE_CAP,
    FEE_RATE,
    HOLD_SIGNAL_THR,
    INITIAL_BALANCE,
    MAX_HOLD_STEPS,
    MSCL_CONSEC_HALT,
    MSCL_HALT_STEPS,
    MSCL_MAX_PENALTY,
    MSCL_ROLL_WIN,
    MULTIPLIER_3SIGMA,
    REWARD_MDD_COEF,
    ROLLING_VOL_WIN,
    TIME_DECAY_LAMBDA,
    TRAILING_PENALTY_COEF,
    V24_4_DualBladeEnv,
)

FUNDING_HOURS_UTC = (0, 8, 16)
FUNDING_PRE_MINUTES = 15
LIQ_REWARD = -100.0
REWARD_CLIP = (-100.0, 10.0)
MAINT_MARGIN_RATE = 0.005
_LOG3 = math.log(3)  # 최대 엔트로피 (3-class uniform)


def _wilder_atr12(df: pd.DataFrame, end_idx: int) -> float:
    sl = max(0, end_idx - 80)
    seg = df.iloc[sl : end_idx + 1]
    if len(seg) < 3:
        return float(df.iloc[end_idx]["close"]) * 0.002
    high = seg["high"].astype(float)
    low = seg["low"].astype(float)
    close = seg["close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=12, adjust=False).mean().iloc[-1]
    return float(max(atr, 1e-12))


def _utc_minute_from_row(row: pd.Series, step_idx: int) -> int:
    ts = row.get("timestamp")
    if ts is not None and not (isinstance(ts, float) and math.isnan(ts)):
        try:
            tsi = int(float(ts))
            if tsi > 1_000_000_000_000:
                tsi //= 1000
            dt = datetime.fromtimestamp(tsi, tz=timezone.utc)
            return dt.hour * 60 + dt.minute
        except Exception:
            pass
    return int((step_idx % CANDLES_PER_DAY) * (24 * 60 // CANDLES_PER_DAY))


def _in_funding_snipe_window(utc_minute: int) -> bool:
    for fh in FUNDING_HOURS_UTC:
        em = fh * 60
        if fh == 0:
            if utc_minute >= 24 * 60 - FUNDING_PRE_MINUTES:
                return True
        else:
            if em - FUNDING_PRE_MINUTES <= utc_minute < em:
                return True
    return False


def _tcn_entropy_ratio(p_up: float, p_down: float, p_chop: float) -> float:
    """TCN 3-class 엔트로피 비율 (0=완전 확신, 1=완전 불확실)."""
    H = 0.0
    for p in (p_up, p_down, p_chop):
        p = max(p, 1e-9)
        H -= p * math.log(p)
    return H / _LOG3


class V25_1_SpecialistEnv(V24_4_DualBladeEnv):
    """
    V25.1 Hybrid Defense — BE·Sliding Barrier·r_sl·Risk Sensor 탑재.
    관측 차원(23) 및 V24.4 피처·마스크 완전 유지.
    """

    def __init__(
        self,
        data_dir: str,
        coin_files: list,
        split_type: str | None = None,
        max_hold: int = MAX_HOLD_STEPS,
        obs_dim: int = 23,
        adx_thr: float = 14.5,
        tcn_gate_thr: float = 0.58,
        mscl_mu: float = 0.004,
        reward_vol_coef: float = 0.05,
        atr_raw_thr: float = 0.0009,
        # ── 레버리지 & 리스크 ──────────────
        leverage: int = 1,
        exchange_max_leverage: int = 10,
        # ── SL / TP / BE ────────────────────
        sl_atr_coef: float = 2.0,
        tp_atr_coef: float = 1.0,
        be_atr_coef: float = 1.5,       # [신규] Break-even 활성화 거리
        # ── 변동성·MDD 페널티 ────────────────
        theta_vol_penalty: float = 0.35,
        phi_mdd_shaping: float = 0.2,
        mm_rate: float = MAINT_MARGIN_RATE,
        # ── Risk Sensor ──────────────────────
        emergency_entropy_thr: float = 0.92,  # [신규] 응급 디레버리지 엔트로피 임계
        post_exit_coef: float = 0.3,          # [신규] 사후 검증 보상 계수
    ):
        # 배율 가드레일: min(exchange_max_leverage, 10)
        _lev_cap = int(max(1, min(exchange_max_leverage, 10)))
        self.leverage = int(max(1, min(_lev_cap, leverage)))
        self.exchange_max_leverage = _lev_cap
        self.sl_atr_coef = float(sl_atr_coef)
        self.tp_atr_coef = float(tp_atr_coef)
        self.be_atr_coef = float(be_atr_coef)
        self.theta_vol_penalty = float(theta_vol_penalty)
        self.phi_mdd_shaping = float(phi_mdd_shaping)
        self.mm_rate = float(mm_rate)
        self.emergency_entropy_thr = float(emergency_entropy_thr)
        self.post_exit_coef = float(post_exit_coef)

        super().__init__(
            data_dir,
            coin_files,
            split_type=split_type,
            max_hold=max_hold,
            obs_dim=obs_dim,
            adx_thr=adx_thr,
            tcn_gate_thr=tcn_gate_thr,
            mscl_mu=mscl_mu,
            reward_vol_coef=reward_vol_coef,
            atr_raw_thr=atr_raw_thr,
        )
        for coin in self.coin_files:
            df = self.cached_dfs[coin]
            if "timestamp" not in df.columns:
                df = df.copy()
                df["timestamp"] = (np.arange(len(df), dtype=np.int64) * 300 + 1_600_000_000) * 1000
                self.cached_dfs[coin] = df

    def _reset_state(self):
        super()._reset_state()
        self._session_peak_equity = float(INITIAL_BALANCE)
        self._post_exit: dict | None = None  # 사후 검증 트래커

    def _liq_prices(self, entry: float, side: str, leverage: float | None = None) -> float:
        L = float(leverage if leverage is not None else self.leverage)
        mm = self.mm_rate
        if side == "long":
            return float(entry * (1.0 - (1.0 - mm) / max(L, 1.0)))
        return float(entry * (1.0 + (1.0 - mm) / max(L, 1.0)))

    def _get_pnl_raw(self) -> float:
        if not self.position:
            return 0.0
        curr = float(self.df.iloc[self.current_step]["close"])
        entry = self.position["entry_price"]
        raw = (curr / entry - 1.0) if self.position["dir"] == "long" else (1.0 - curr / entry)
        lev = float(self.position.get("leverage", self.leverage))
        return raw * self.position["size"] * lev

    def _get_pnl(self) -> float:
        return self._get_pnl_raw()

    def _compute_mscl_penalty(self, trade_won: bool) -> float:
        return super()._compute_mscl_penalty(trade_won)

    def _close_position(self) -> tuple:
        pnl_raw = self._get_pnl_raw()
        lev = float(self.position.get("leverage", self.leverage))
        fee_mult = min(lev, 10.0)
        net_pnl = pnl_raw - self.position["size"] * FEE_RATE * fee_mult
        self.balance *= 1.0 + net_pnl
        self.balance = max(self.balance, 0.0)

        won = net_pnl > 0
        self.total_trades += 1
        if won:
            self.winning_trades += 1

        mscl_penalty = self._compute_mscl_penalty(won)

        pure_return = pnl_raw / max(self.position["size"] * lev, 1e-12)
        delta = float(self.df["delta_thresh"].iloc[self.current_step])

        if pure_return > 0.02:
            realized_r = net_pnl * 3.0
        elif net_pnl >= 3 * delta:
            realized_r = net_pnl * MULTIPLIER_3SIGMA
        elif net_pnl < 0:
            realized_r = net_pnl * 1.2
        else:
            realized_r = net_pnl

        self.position = None
        self.running_max_equity = 0.0
        return realized_r, won, mscl_penalty

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        reward = 0.0
        done = False
        info: dict = {}

        prev_equity = self._get_equity()

        # ── MSCL 할트 처리 ────────────────────────────────────────────
        if self.mscl_halt:
            self.mscl_halt_timer -= 1
            if self.mscl_halt_timer <= 0:
                self.mscl_halt = False
                self.consecutive_count = 0
                self.last_signal_dir = 0

        self.steps_today += 1
        if self.steps_today >= CANDLES_PER_DAY:
            self.steps_today = 0
            self.daily_trade_count = 0

        row = self.df.iloc[self.current_step]
        hi = float(row["high"])
        lo = float(row["low"])
        close_now = float(row["close"])

        adx_14 = float(row.get("adx_14", 30.0))
        atr_raw = float(row.get("atr_raw", 0.002))
        atr_z = float(row.get("atr_z", 0.0))

        if adx_14 < self.adx_thr or atr_raw < self.atr_raw_thr:
            action_val = 0.0

        signal_dir = 0
        if action_val > HOLD_SIGNAL_THR:
            signal_dir = +1
        elif action_val < -HOLD_SIGNAL_THR:
            signal_dir = -1

        p_up = float(row.get("tcn_p_up", 1 / 3))
        p_chop = float(row.get("tcn_p_chop", 1 / 3))
        p_down = float(row.get("tcn_p_down", 1 / 3))
        tcn_p_var = float(row.get("tcn_p_var", 0.0))

        # ── [Risk Sensor] TCN 엔트로피 계산 → 응급 디레버리지 판단 ───
        entropy_ratio = _tcn_entropy_ratio(p_up, p_down, p_chop)
        _emergency = entropy_ratio > self.emergency_entropy_thr
        # ──────────────────────────────────────────────────────────────

        if signal_dir == +1 and p_up < self.tcn_gate_thr:
            signal_dir = 0
            self.consecutive_count = 0
            self.last_signal_dir = 0
        elif signal_dir == -1 and p_down < self.tcn_gate_thr:
            signal_dir = 0
            self.consecutive_count = 0
            self.last_signal_dir = 0

        funding_rate = float(row.get("funding_rate", 0.0))
        self.momentum_signal = p_up - p_down

        if signal_dir != 0 and signal_dir == self.last_signal_dir:
            self.consecutive_count += 1
        elif signal_dir != 0:
            self.consecutive_count = 1
            self.last_signal_dir = signal_dir
        else:
            self.consecutive_count = 0
            self.last_signal_dir = 0

        can_enter = (
            self.consecutive_count >= CONSECUTIVE_NEED
            and self.daily_trade_count < DAILY_TRADE_CAP
            and not self.position
            and not self.mscl_halt
        )

        realized_reward_closed = 0.0
        atr12 = _wilder_atr12(self.df, self.current_step)
        utc_min = _utc_minute_from_row(row, self.current_step)
        fund_window = _in_funding_snipe_window(utc_min)

        # ── 포지션 관리: LIQ → BE → SL → Trail TP ──────────────────
        if self.position:
            liq_p = float(self.position["liq_price"])
            side = self.position["dir"]

            # 1) 강제 청산 (Liquidation)
            if side == "long" and lo <= liq_p:
                self.position = None
                self.balance = max(self.balance * 0.02, 0.0)
                done = True
                info = {**self._get_episode_info(), "liquidated": True}
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                return obs, float(np.clip(LIQ_REWARD, REWARD_CLIP[0], REWARD_CLIP[1])), done, False, info
            if side == "short" and hi >= liq_p:
                self.position = None
                self.balance = max(self.balance * 0.02, 0.0)
                done = True
                info = {**self._get_episode_info(), "liquidated": True}
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                return obs, float(np.clip(LIQ_REWARD, REWARD_CLIP[0], REWARD_CLIP[1])), done, False, info

            entry_p = float(self.position["entry_price"])

            # 2) [신규] Break-even 활성화
            if not self.position.get("be_activated", False):
                be_dist = self.be_atr_coef * atr12
                if side == "long" and hi >= entry_p + be_dist:
                    self.position["sl_price"] = entry_p  # SL → 본절
                    self.position["be_activated"] = True
                    reward += 0.002  # BE 달성 소액 보너스
                elif side == "short" and lo <= entry_p - be_dist:
                    self.position["sl_price"] = entry_p
                    self.position["be_activated"] = True
                    reward += 0.002

            # 3) [응급] Emergency Deleveraging — SL 30% 타이트 조정
            if _emergency:
                sl_curr = float(self.position["sl_price"])
                if side == "long":
                    gap = close_now - sl_curr
                    if gap > 0:
                        self.position["sl_price"] = sl_curr + gap * 0.3
                else:
                    gap = sl_curr - close_now
                    if gap > 0:
                        self.position["sl_price"] = sl_curr - gap * 0.3
                reward -= 0.001  # 위험 구간 소액 페널티

            # 4) SL 체크 (BE로 이동되었을 수 있음)
            sl_p = float(self.position["sl_price"])
            if side == "long" and lo <= sl_p:
                self._post_exit = {
                    "dir": side,
                    "exit_price": close_now,
                    "steps_left": 5,
                }
                realized_r, won, mscl_pen = self._close_position()
                reward -= mscl_pen
                realized_reward_closed = realized_r

            elif side == "short" and hi >= sl_p:
                self._post_exit = {
                    "dir": side,
                    "exit_price": close_now,
                    "steps_left": 5,
                }
                realized_r, won, mscl_pen = self._close_position()
                reward -= mscl_pen
                realized_reward_closed = realized_r

            elif self.position:
                # 5) Sliding Trailing TP (BE 이후 강화)
                act_dist = self.tp_atr_coef * atr12
                if side == "long":
                    if hi >= entry_p + act_dist:
                        self.position["trail_active"] = True
                    if self.position.get("trail_active"):
                        self.position["peak_track"] = max(
                            float(self.position.get("peak_track", entry_p)), hi
                        )
                        trail = float(self.position["peak_track"]) - self.tp_atr_coef * atr12
                        if lo <= trail:
                            self._post_exit = {
                                "dir": side,
                                "exit_price": close_now,
                                "steps_left": 5,
                            }
                            realized_r, won, mscl_pen = self._close_position()
                            reward -= mscl_pen
                            realized_reward_closed = realized_r
                else:
                    if lo <= entry_p - act_dist:
                        self.position["trail_active"] = True
                    if self.position.get("trail_active"):
                        self.position["peak_track"] = min(
                            float(self.position.get("peak_track", entry_p)), lo
                        )
                        trail = float(self.position["peak_track"]) + self.tp_atr_coef * atr12
                        if hi >= trail:
                            self._post_exit = {
                                "dir": side,
                                "exit_price": close_now,
                                "steps_left": 5,
                            }
                            realized_r, won, mscl_pen = self._close_position()
                            reward -= mscl_pen
                            realized_reward_closed = realized_r

        # ── 신규 진입 ────────────────────────────────────────────────
        if not self.position and not done:
            if can_enter and signal_dir != 0:
                base_size = abs(action_val)
                scaled_size = base_size * max(0.01, (1.0 - tcn_p_var * 4.0))
                if atr_z > 2.0:
                    scaled_size *= 0.5
                reward += 0.0001

                dir_ = "long" if signal_dir == 1 else "short"
                entry_p = close_now

                # 응급 시 leverage = 1 강제
                eff_leverage = 1.0 if _emergency else float(self.leverage)
                sl_off = self.sl_atr_coef * atr12
                if _emergency:
                    sl_off *= 0.7  # SL 30% 타이트

                sl_price = (entry_p - sl_off) if dir_ == "long" else (entry_p + sl_off)

                self.position = {
                    "dir": dir_,
                    "entry_price": entry_p,
                    "size": scaled_size,
                    "duration": 0,
                    "leverage": eff_leverage,
                    "liq_price": self._liq_prices(entry_p, dir_, eff_leverage),
                    "sl_price": sl_price,
                    "entry_atr": atr12,
                    "trail_active": False,
                    "peak_track": entry_p,
                    "be_activated": False,  # [신규]
                }
                fee_mult = min(eff_leverage, 10.0)
                self.balance -= self.balance * scaled_size * FEE_RATE * fee_mult
                self.daily_trade_count += 1
                self.consecutive_count = 0
                self.running_max_equity = prev_equity

        # ── 포지션 보유 중 처리 ──────────────────────────────────────
        if self.position and not done:
            self.position["duration"] += 1
            self.running_max_equity = max(self.running_max_equity, prev_equity)

            trailing_penalty = TRAILING_PENALTY_COEF * max(0, self.running_max_equity - prev_equity)
            reward -= trailing_penalty / INITIAL_BALANCE

            decay_penalty = self.position["duration"] * TIME_DECAY_LAMBDA * self.position["size"]
            reward -= decay_penalty

            base_f = 0.3 * funding_rate * self.position["size"]
            if self.position["dir"] == "long":
                funding_reward = -base_f
            else:
                funding_reward = base_f
            recv = (self.position["dir"] == "short" and funding_rate > 0) or (
                self.position["dir"] == "long" and funding_rate < 0
            )
            if fund_window:
                funding_reward *= 1.6 if recv else 0.75
            reward += funding_reward

            pos_sign = 1 if self.position["dir"] == "long" else -1
            should_close = (
                (signal_dir != 0 and signal_dir != pos_sign)
                or (abs(action_val) < HOLD_SIGNAL_THR)
                or (self.position["duration"] >= self.max_hold)
            )
            if should_close:
                self._post_exit = {
                    "dir": self.position["dir"],
                    "exit_price": close_now,
                    "steps_left": 5,
                }
                realized_r, won, mscl_pen = self._close_position()
                reward -= mscl_pen
                realized_reward_closed = realized_r

        # ── [신규] 사후 검증 보상 r_sl (EXIT 후 5봉 추적) ──────────
        if self._post_exit is not None and self._post_exit["steps_left"] > 0:
            curr_close = float(self.df.iloc[self.current_step]["close"])
            exit_p = self._post_exit["exit_price"]
            pdir = self._post_exit["dir"]
            move = (curr_close - exit_p) / max(exit_p, 1e-12)
            # Long 청산 후: 가격 하락 = Good Exit (+), 반등 = Bad Exit (-)
            # Short 청산 후: 가격 상승 = Good Exit (+), 하락 = Bad Exit (-)
            r_sl = (-move if pdir == "long" else move) * self.post_exit_coef
            reward += r_sl
            self._post_exit["steps_left"] -= 1
            if self._post_exit["steps_left"] <= 0:
                self._post_exit = None
        # ─────────────────────────────────────────────────────────────

        new_equity = self._get_equity()
        self._session_peak_equity = max(self._session_peak_equity, new_equity)
        delta_equity = (new_equity - prev_equity) / INITIAL_BALANCE

        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        current_dd = (self.peak_equity - new_equity) / (self.peak_equity + 1e-9)
        if current_dd > self.mdd:
            self.mdd = current_dd

        dd_sess = (self._session_peak_equity - new_equity) / (self._session_peak_equity + 1e-9)
        reward -= self.phi_mdd_shaping * max(0.0, dd_sess)

        self.recent_returns.append(delta_equity)
        if len(self.recent_returns) > ROLLING_VOL_WIN:
            self.recent_returns.pop(0)
        vol = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0
        var_r = float(np.var(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0
        reward -= self.theta_vol_penalty * var_r

        base_reward = delta_equity - REWARD_MDD_COEF * current_dd - self.reward_vol_coef * vol
        reward += base_reward

        expert_r = self._compute_srddqn_reward(base_reward, realized_pnl=realized_reward_closed)
        if expert_r > base_reward:
            reward += expert_r - base_reward

        self.total_steps += 1
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            if self.position:
                _, _, mscl_pen = self._close_position()
                reward -= mscl_pen
            done = True
            info = self._get_episode_info()

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(np.clip(reward, REWARD_CLIP[0], REWARD_CLIP[1])), done, False, info
