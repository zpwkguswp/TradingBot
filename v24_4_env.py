"""
V24 Sovereign Dual-Blade — Hybrid Environment (Stage 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 구현 로직:
  1. 단일 스케일 5m 피처 사용 (차원 축소)
  2. 계단식 체제 전환 (Tiered Regime Switching): atr_z 기반 비상 제동 플랜 적용
  3. MSCL 보상 압력 고도화 (abs 차이 활용)
  4. Optuna용 하이퍼파라미터 인젝션 지원
"""
import os
import math
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────────────────
# 기본 상수
# ─────────────────────────────────────────────────────────
FEE_RATE           = 0.0006    
INITIAL_BALANCE    = 10_000.0
HOLD_SIGNAL_THR    = 0.15      
CONSECUTIVE_NEED   = 3         
DAILY_TRADE_CAP    = 12        
CANDLES_PER_DAY    = 288       
MAX_HOLD_STEPS     = 288       

REWARD_MDD_COEF    = 0.2       
REWARD_VOL_COEF    = 0.05      
FUNDING_REWARD_W   = 0.3       

EXPERT_WINDOW      = 20        
MSCL_MU            = 0.004      # [V24.1] 페널티 대폭 완화 (기조 0.04 -> 0.004)
MSCL_ROLL_WIN      = 20        
MSCL_CONSEC_HALT   = 3         
MSCL_HALT_STEPS    = 24        
MSCL_MAX_PENALTY   = 0.002      # [V24.1] 페널티 캡 (Cap) 설정으로 공포심 제한

TIME_DECAY_LAMBDA  = 0.00005   
ROLLING_VOL_WIN    = 20
TRAILING_PENALTY_COEF = 0.2
MULTIPLIER_3SIGMA = 3.0        

class V24_4_DualBladeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_dir: str, coin_files: list,
                 split_type: str = None,
                 max_hold: int = MAX_HOLD_STEPS,
                 obs_dim: int = 23,
                 # [V24.4 Optima Params]
                 adx_thr: float = 14.5, 
                 tcn_gate_thr: float = 0.58,
                 mscl_mu: float = 0.004,
                 reward_vol_coef: float = 0.05,
                 atr_raw_thr: float = 0.0009):
                 
        super().__init__()
        self.data_dir   = data_dir
        self.coin_files = coin_files
        self.split_type = split_type
        self.max_hold   = max_hold

        self.adx_thr = adx_thr
        self.tcn_gate_thr = tcn_gate_thr
        self.mscl_mu = mscl_mu
        self.reward_vol_coef = reward_vol_coef
        self.atr_raw_thr = atr_raw_thr

        self.cached_dfs = {}
        for coin in self.coin_files:
            path = os.path.join(data_dir, coin)
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                # Fallback to parquet if no extension matches (legacy)
                df = pd.read_parquet(path)
            
            # 결측치 폴백
            for col, val in [('tcn_p_down', 1/3), ('tcn_p_chop', 1/3), ('tcn_p_up', 1/3),
                             ('tcn_p_var', 0.0), ('adx_14', 30.0), ('atr_raw', 0.002), ('delta_thresh', 0.001)]:
                if col not in df.columns:
                    df[col] = val

            df = df.dropna(subset=['tcn_p_up', 'adx_14']).reset_index(drop=True)
            
            if split_type in ('train', 'eval') and len(df) > 0:
                split_idx = int(len(df) * 0.8)
                df = df.iloc[:split_idx] if split_type == 'train' else df.iloc[split_idx:]
                df = df.reset_index(drop=True)
            self.cached_dfs[coin] = df

        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self.obs_dim),), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._reset_state()

    def _reset_state(self):
        self.balance       = INITIAL_BALANCE
        self.peak_equity   = INITIAL_BALANCE
        self.position      = None
        self.current_step  = 0
        self.total_steps   = 0
        self.total_trades  = 0
        self.winning_trades= 0
        self.mdd           = 0.0
        self.last_equity   = INITIAL_BALANCE
        self.recent_returns= []

        self.consecutive_count = 0
        self.last_signal_dir   = 0
        self.daily_trade_count = 0
        self.steps_today       = 0

        self.expert_reward_buf = []    
        self.momentum_signal   = 0.0   

        self.trade_outcomes    = []    
        self.consec_loss_count = 0     
        self.mscl_halt         = False 
        self.mscl_halt_timer   = 0     
        self.mscl_halt_count   = 0     

        self.n_win_total  = 0
        self.n_loss_total = 0
        
        self.running_max_equity = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_coin = random.choice(self.coin_files)
        self.df = self.cached_dfs[self.current_coin]
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        p_down = float(row.get('tcn_p_down', 1/3))
        p_chop = float(row.get('tcn_p_chop', 1/3))
        p_up   = float(row.get('tcn_p_up',   1/3))
        
        coint_div = float(row.get('coint_div', 0.0))
        tfi_5m  = float(row.get('tfi_5m',  0.0))
        crsi_5m = float(row.get('crsi_5m', 0.5))
        rsi_z   = float(row.get('rsi_z', 0.0))
        atr_z   = float(row.get('atr_z', 0.0))
        vol_t   = float(row.get('vol_tanh', 0.0))
        price_z = float(row.get('price_zscore', 0.0))

        curr_equity = self._get_equity()
        pnl         = self._get_pnl()

        pos_dir  = 0.0
        pos_size = 0.0
        dur_r    = 0.0
        if self.position:
            pos_dir  = 1.0 if self.position['dir'] == 'long' else -1.0
            pos_size = self.position['size']
            dur_r    = self.position['duration'] / self.max_hold

        consec_r      = min(self.consecutive_count / CONSECUTIVE_NEED, 2.0)
        daily_trade_r = min(self.daily_trade_count / DAILY_TRADE_CAP, 2.0)
        funding_z     = float(row.get('funding_rate_z', 0.0))
        mscl_halt_f   = 1.0 if self.mscl_halt else 0.0
        
        # V24 Features
        tcn_p_var = float(row.get('tcn_p_var', 0.0))
        adx = float(row.get('adx_14', 25.0)) / 100.0   
        atr_raw_sc = float(row.get('atr_raw', 0.002)) * 100.0 

        # V24: 23 features total 
        return np.array([
            p_down, p_chop, p_up,         
            coint_div,                    
            tfi_5m, crsi_5m,  
            rsi_z, atr_z, vol_t, price_z, 
            curr_equity / INITIAL_BALANCE,
            pnl,                          
            pos_dir, pos_size, dur_r,     
            consec_r, daily_trade_r,      
            funding_z, mscl_halt_f,
            tcn_p_var, adx, atr_raw_sc,   
            (self.running_max_equity - curr_equity) / INITIAL_BALANCE if self.position else 0.0
        ], dtype=np.float32)

    def _get_pnl(self) -> float:
        if not self.position: return 0.0
        curr  = float(self.df.iloc[self.current_step]['close'])
        entry = self.position['entry_price']
        raw   = (curr / entry - 1.0) if self.position['dir'] == 'long' else (1.0 - curr / entry)
        return raw * self.position['size']

    def _get_pnl_raw(self) -> float:
        if not self.position: return 0.0
        curr  = float(self.df.iloc[self.current_step]['close'])
        entry = self.position['entry_price']
        raw   = (curr / entry - 1.0) if self.position['dir'] == 'long' else (1.0 - curr / entry)
        return raw * self.position['size']

    def _get_equity(self) -> float:
        return self.balance * (1 + self._get_pnl()) if self.position else self.balance

    def _compute_srddqn_reward(self, base_reward: float, realized_pnl: float = 0.0) -> float:
        self.expert_reward_buf.append(base_reward)
        if len(self.expert_reward_buf) > EXPERT_WINDOW:
            self.expert_reward_buf.pop(0)
        expert_r = float(np.mean(self.expert_reward_buf)) if self.expert_reward_buf else 0.0
        predicted_r = realized_pnl if realized_pnl > 0 else 0.0
        return max(expert_r, predicted_r)

    def _compute_mscl_penalty(self, trade_won: bool) -> float:
        outcome = +1 if trade_won else -1
        self.trade_outcomes.append(outcome)
        if len(self.trade_outcomes) > MSCL_ROLL_WIN:
            self.trade_outcomes.pop(0)

        n_win  = sum(1 for o in self.trade_outcomes if o == +1)
        n_loss = sum(1 for o in self.trade_outcomes if o == -1)

        r_port      = self.balance / INITIAL_BALANCE
        # [V24] 절대값 차이를 사용하여 패널티 효율 극대화
        diff        = float(abs(n_loss - n_win))
        raw_penalty = self.mscl_mu * diff * r_port
        penalty     = min(raw_penalty, MSCL_MAX_PENALTY)

        if not trade_won:
            self.consec_loss_count += 1
            self.n_loss_total      += 1
        else:
            self.consec_loss_count  = 0
            self.n_win_total       += 1

        if self.consec_loss_count >= MSCL_CONSEC_HALT:
            self.mscl_halt         = True
            self.mscl_halt_timer   = MSCL_HALT_STEPS
            self.consec_loss_count = 0
            self.mscl_halt_count  += 1

        return penalty

    def _close_position(self) -> tuple:
        pnl_raw   = self._get_pnl_raw()
        net_pnl   = pnl_raw - self.position['size'] * FEE_RATE
        self.balance *= (1 + net_pnl)
        self.balance  = max(self.balance, 0.0)

        won = net_pnl > 0
        self.total_trades += 1
        if won: self.winning_trades += 1

        mscl_penalty = self._compute_mscl_penalty(won)
        
        pure_return = (pnl_raw / self.position['size']) if self.position['size'] > 0 else 0.0
        delta = float(self.df['delta_thresh'].iloc[self.current_step])
        
        # [V24.2] 홈런 보상 (Reward Shaping)
        if pure_return > 0.02: realized_r = net_pnl * 3.0
        elif net_pnl >= 3 * delta: realized_r = net_pnl * MULTIPLIER_3SIGMA
        elif net_pnl < 0: realized_r = net_pnl * 1.2
        else: realized_r = net_pnl
            
        self.position = None
        self.running_max_equity = 0.0 
        return realized_r, won, mscl_penalty

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        reward     = 0.0
        done       = False
        info       = {}

        prev_equity = self._get_equity()

        if self.mscl_halt:
            self.mscl_halt_timer -= 1
            if self.mscl_halt_timer <= 0:
                self.mscl_halt = False
                self.consecutive_count = 0
                self.last_signal_dir   = 0

        self.steps_today += 1
        if self.steps_today >= CANDLES_PER_DAY:
            self.steps_today       = 0
            self.daily_trade_count = 0

        row = self.df.iloc[self.current_step]
        
        adx_14 = float(row.get('adx_14', 30.0))
        atr_raw = float(row.get('atr_raw', 0.002))
        atr_z = float(row.get('atr_z', 0.0))
        
        # Action Masking (Optuna 튜닝된 파라미터 적용)
        if adx_14 < self.adx_thr or atr_raw < self.atr_raw_thr:
            action_val = 0.0 # Force Hold

        signal_dir = 0
        if   action_val >  HOLD_SIGNAL_THR: signal_dir = +1
        elif action_val < -HOLD_SIGNAL_THR: signal_dir = -1

        p_up   = float(row.get('tcn_p_up',   1/3))
        p_chop = float(row.get('tcn_p_chop', 1/3))
        p_down = float(row.get('tcn_p_down', 1/3))
        tcn_p_var = float(row.get('tcn_p_var', 0.0))
        
        # TCN 신뢰도 게이트
        if signal_dir == +1 and p_up < self.tcn_gate_thr:
            signal_dir = 0
            self.consecutive_count = 0; self.last_signal_dir = 0
        elif signal_dir == -1 and p_down < self.tcn_gate_thr:
            signal_dir = 0
            self.consecutive_count = 0; self.last_signal_dir = 0

        funding_rate = float(row.get('funding_rate', 0.0))
        self.momentum_signal = p_up - p_down

        if signal_dir != 0 and signal_dir == self.last_signal_dir:
            self.consecutive_count += 1
        elif signal_dir != 0:
            self.consecutive_count  = 1
            self.last_signal_dir    = signal_dir
        else:
            self.consecutive_count  = 0
            self.last_signal_dir    = 0

        can_enter = (
            self.consecutive_count >= CONSECUTIVE_NEED and
            self.daily_trade_count  < DAILY_TRADE_CAP  and
            not self.position and
            not self.mscl_halt
        )

        realized_reward_closed = 0.0
        
        # 진입 로직
        if not self.position:
            if can_enter and signal_dir != 0:
                base_size = abs(action_val)
                scaled_size = base_size * max(0.01, (1.0 - tcn_p_var * 4.0)) 
                
                # [V24.3] Tiered Regime Switching 조율 (atr_z > 2.0)
                if atr_z > 2.0:
                    scaled_size *= 0.5
                    
                # [V24.1] '근거 있는 진입'에 대한 용기 보상 (Action Reward)
                reward += 0.0001 
                    
                dir_      = 'long' if signal_dir == 1 else 'short'
                entry_p   = float(self.df.iloc[self.current_step]['close'])
                self.position = {
                    'dir':         dir_,
                    'entry_price': entry_p,
                    'size':        scaled_size,
                    'duration':    0,
                }
                self.balance -= self.balance * scaled_size * FEE_RATE
                self.daily_trade_count += 1
                self.consecutive_count  = 0
                self.running_max_equity = prev_equity 

        # 보유 중 로직
        elif self.position:
            self.position['duration'] += 1
            self.running_max_equity = max(self.running_max_equity, prev_equity)
            
            # [V24] 트레일링 보상
            net_w_s = prev_equity
            trailing_penalty = TRAILING_PENALTY_COEF * max(0, self.running_max_equity - net_w_s)
            reward -= (trailing_penalty / INITIAL_BALANCE)

            decay_penalty = self.position['duration'] * TIME_DECAY_LAMBDA * self.position['size']
            reward -= decay_penalty

            # 펀딩비 보상
            if self.position['dir'] == 'short': funding_reward = FUNDING_REWARD_W * funding_rate * self.position['size']
            else: funding_reward = -FUNDING_REWARD_W * funding_rate * self.position['size']
            reward += funding_reward

            pos_sign = 1 if self.position['dir'] == 'long' else -1
            should_close = (
                (signal_dir != 0 and signal_dir != pos_sign) or
                (abs(action_val) < HOLD_SIGNAL_THR) or
                (self.position['duration'] >= self.max_hold)
            )
            if should_close:
                realized_r, won, mscl_pen = self._close_position()
                reward -= mscl_pen   
                realized_reward_closed = realized_r

        new_equity   = self._get_equity()
        delta_equity = (new_equity - prev_equity) / INITIAL_BALANCE

        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        current_dd = (self.peak_equity - new_equity) / (self.peak_equity + 1e-9)
        if current_dd > self.mdd:
            self.mdd = current_dd

        self.recent_returns.append(delta_equity)
        if len(self.recent_returns) > ROLLING_VOL_WIN:
            self.recent_returns.pop(0)
        vol = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0

        base_reward = delta_equity - REWARD_MDD_COEF * current_dd - self.reward_vol_coef * vol
        reward     += base_reward

        srddqn_r  = self._compute_srddqn_reward(base_reward, realized_pnl=realized_reward_closed)
        if srddqn_r > base_reward: reward += (srddqn_r - base_reward)

        self.total_steps  += 1
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            if self.position:
                net_pnl, won, mscl_pen = self._close_position()
                reward -= mscl_pen
            done = True
            info = self._get_episode_info()

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(np.clip(reward, -10.0, 10.0)), done, False, info

    def _get_episode_info(self) -> dict:
        win_rate   = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        final_ret  = (self.balance / INITIAL_BALANCE) - 1.0
        std_r      = float(np.std(self.recent_returns))  if len(self.recent_returns) > 1 else 0.0
        mean_r     = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        sharpe     = (mean_r / (std_r + 1e-9)) * math.sqrt(CANDLES_PER_DAY * 365)
        days       = max(self.total_steps / CANDLES_PER_DAY, 1)

        return {
            'episode_finished':  True,
            'bot_return':        final_ret,
            'mdd':               self.mdd,
            'total_trades':      self.total_trades,
            'win_rate':          win_rate,
            'sharpe_ratio':      sharpe,
            'daily_trade_avg':   self.total_trades / days,
            'n_win':             self.n_win_total,
            'n_loss':            self.n_loss_total,
            'mscl_halt_count':   self.mscl_halt_count,  
        }
