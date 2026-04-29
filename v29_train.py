import os
import gc
import json
import time
import subprocess
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
import pynvml
import warnings
import multiprocessing
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

# Windows subproc safety (Must be called before any multiprocessing creates processes)
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from v29_env import V29_Universal_Env

# [V29 Phase 1] Universal Alpha MODE
MODE = "TUNE" # "TUNE" or "TRAIN"

DATA_DIR          = "data_storage"
ELITE_DIR         = "elite_weights"
LOG_DIR           = "v29_logs"
STUDY_DB          = "sqlite:///v29_overseer.db"
STUDY_NAME        = "v29_universal_alpha_v1.0"
TRADE_LOG_FILE    = "v29_trade_log.csv"

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT", "NEARUSDT"]

os.makedirs(ELITE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

N_STACK = 4
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

class SanityCheckCallback(BaseCallback):
    CHECK_INTERVAL  = 50_000
    PRUNE_STEP      = 50_000
    PRUNE_GOAL      = 3
    BANKRUPT_RATIO  = 0.60

    def __init__(self, timeframe: str, initial_balance: float = 10_000.0, verbose=0):
        super().__init__(verbose)
        self.timeframe         = timeframe
        self._initial_balance  = initial_balance
        self._last_check_step  = int(0)
        self.pruned            = False
        self.trade_logs        = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos: return True

        for info in infos:
            if info.get("trade_closed"):
                log_entry = {
                    "Timeframe": self.timeframe,
                    "Entry_Price": info.get("trade_Entry_Price"),
                    "Exit_Price": info.get("trade_Exit_Price"),
                    "Actual_PnL": info.get("trade_Actual_PnL"),
                    "MFE": info.get("trade_MFE"),
                    "MAE": info.get("trade_MAE"),
                    "Entry_Disparity": info.get("trade_Entry_Disparity"),
                    "Exit_Reason": info.get("trade_Exit_Reason")
                }
                self.trade_logs.append(log_entry)

        balances = [info.get("balance", self._initial_balance) for info in infos]
        avg_balance = float(np.mean(balances))
        if avg_balance < self._initial_balance * self.BANKRUPT_RATIO:
            print(f"[Sanity] PRUNE: Step {self.num_timesteps} | Avg Balance {avg_balance:.0f} hit limit.")
            self.pruned = True
            return False

        if self.num_timesteps >= self.PRUNE_STEP and self.num_timesteps < self.PRUNE_STEP + 1000:
            total_goal_hits = sum(info.get("lifetime_goal_hits", 0) for info in infos)
            if total_goal_hits < self.PRUNE_GOAL:
                print(f"[Sanity] FAST PRUNE: Step {self.num_timesteps} | Goal Hits {total_goal_hits} < {self.PRUNE_GOAL}.")
                self.pruned = True
                return False

        if self.num_timesteps - self._last_check_step >= self.CHECK_INTERVAL:
            self._last_check_step = self.num_timesteps
            self.save_trade_logs()
            
        return True

    def save_trade_logs(self):
        if not self.trade_logs: return
        df = pd.DataFrame(self.trade_logs)
        file_exists = os.path.isfile(TRADE_LOG_FILE)
        df.to_csv(TRADE_LOG_FILE, mode='a', index=False, header=not file_exists)
        self.trade_logs = []

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]

class TCN6LayerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # observation_space represents flattened VecFrameStack of shape (n_stack * 37,)
        super().__init__(observation_space, features_dim)
        
        self.obs_dim = 36 # 36 dims for TCN
        self.coin_dim = 1 # 1 dim for embedding
        self.total_dim = self.obs_dim + self.coin_dim
        
        self.tcn = nn.Sequential(
            CausalConv1d(self.obs_dim, 16, 12, 1), nn.ReLU(), nn.BatchNorm1d(16),
            CausalConv1d(16, 32, 8, 2), nn.ReLU(), nn.BatchNorm1d(32),
            CausalConv1d(32, 64, 5, 4), nn.ReLU(), nn.BatchNorm1d(64),
            CausalConv1d(64, 128, 3, 8), nn.ReLU(), nn.BatchNorm1d(128),
            CausalConv1d(128, 256, 3, 16), nn.ReLU(), nn.BatchNorm1d(256),
            CausalConv1d(256, 256, 2, 32), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.coin_embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        
        # 256 from TCN + 16 from Coin Embedding = 272
        self.fc = nn.Linear(256 + 16, features_dim)

    def forward(self, observations):
        n_stack = 4
        # Reshape to (batch_size, n_stack, 37)
        x_unrolled = observations.view(-1, n_stack, self.total_dim)
        
        # 1. TCN Pathway
        x_features = x_unrolled[:, :, :self.obs_dim].transpose(1, 2) # (batch, 36, n_stack)
        tcn_out = self.tcn(x_features)
        
        # 2. Embedding Pathway
        # Since coin_id is static across stack frames, just grab the first frame's coin_id
        coin_idx = x_unrolled[:, 0, self.obs_dim].long() # Shape (batch,)
        emb_out = self.coin_embedding(coin_idx)
        
        # 3. Concatenate and pass to FC
        combined = th.cat([tcn_out, emb_out], dim=1) # (batch, 272)
        return self.fc(combined)

def make_env(ticker: str, tf: str, split: str, params: dict):
    coin_id = TICKERS.index(ticker)
    def _init():
        return V29_Universal_Env(
            data_dir=DATA_DIR, 
            coin_files=[f"{ticker}_{tf}.parquet"], 
            coin_id=coin_id,
            split_type=split,
            target_profit=0.008,
            far_th=params.get("far_th", 2.5),
            sl_atr_coef=params.get("sl_atr_coef", 3.2),
            trail_act=params.get("trail_act", 0.0)
        )
    return _init

def objective(trial):
    tf = trial.suggest_categorical("timeframe", ["1h", "2h"])
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    ent = trial.suggest_float("ent_coef", 0.001, 0.05)
    far_th = trial.suggest_float("far_th", 2.0, 5.0)
    sl_atr = trial.suggest_float("sl_atr_coef", 2.0, 4.0)
    
    params = {"far_th": far_th, "sl_atr_coef": sl_atr}
    
    # [V29] 10 Parallel Universal Environments
    venv = SubprocVecEnv([make_env(t, tf, "train", params) for t in TICKERS])
    venv = VecFrameStack(venv, n_stack=N_STACK)
    
    policy_kwargs = dict(
        features_extractor_class=TCN6LayerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128])
    )
    
    model = PPO("MlpPolicy", venv, verbose=0, policy_kwargs=policy_kwargs,
                learning_rate=lr, ent_coef=ent, device=DEVICE)
                
    steps = 150_000 if MODE == "TUNE" else 2_000_000
    callback = SanityCheckCallback(timeframe=tf)
    
    model.learn(total_timesteps=steps, callback=callback)
    callback.save_trade_logs()
    
    if callback.pruned:
        return -1.0

    # [V29] Extreme Eval Loop for measuring all 10 coins
    eval_venv = SubprocVecEnv([make_env(t, tf, "eval", params) for t in TICKERS])
    eval_venv = VecFrameStack(eval_venv, n_stack=N_STACK)
    obs = eval_venv.reset()
    
    episodes_done = np.zeros(len(TICKERS))
    total_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    bot_rets = np.zeros(len(TICKERS))
    peak_equities = np.ones(len(TICKERS))
    mdds = np.zeros(len(TICKERS))
    
    while not (episodes_done >= 1).all():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, step_dones, infos = eval_venv.step(action)
        
        for i, info in enumerate(infos):
            # Record single full pass cleanly
            if episodes_done[i] >= 1: 
                continue
                
            if step_dones[i]:
                episodes_done[i] += 1
                
            if info.get("trade_closed"):
                total_trades += 1
                trade_pnl = info.get("trade_Actual_PnL", 0.0)
                if trade_pnl > 0: gross_profit += trade_pnl
                else: gross_loss += abs(trade_pnl)
            
            equity = info.get("bot_equity", 1.0)
            if equity > peak_equities[i]:
                peak_equities[i] = equity
            dd = (peak_equities[i] - equity) / (peak_equities[i] + 1e-9)
            if dd > mdds[i]:
                mdds[i] = dd
                
            bot_rets[i] = equity - 1.0
            
    pf = gross_profit / (gross_loss + 1e-9)
    avg_ret = bot_rets.mean()
    avg_mdd = mdds.mean()
    
    trial.set_user_attr("pf", float(pf))
    trial.set_user_attr("avg_ret", float(avg_ret))
    trial.set_user_attr("mdd", float(avg_mdd))
    trial.set_user_attr("trades", int(total_trades))
    
    # === [긴급 수술: V28 황금 밸런스 공식의 부활] ===
    if total_trades < 12: 
        score = -2.0  # 더 엄격하게 처벌 (기아 상태 방지)
    else:
        # 🌟 통계적 신뢰도 가중치 계산 (40회 기준 1.0)
        confidence_weight = np.log(total_trades) / np.log(40.0)
        
        # 🌟 기본 밸런스 점수 계산 (MDD 페널티 -2.0 부활!)
        # (주의: V29에서는 10개 코인의 평균인 avg_ret과 avg_mdd를 사용합니다)
        base_score = ((pf - 1.0) * 0.4) + (avg_ret * 0.2) - (avg_mdd * 2.0)
        
        # 가중치 적용 (양수일 때만 보상 강화)
        if base_score > 0:
            score = base_score * confidence_weight
        else:
            score = base_score
    # ===============================================
        
    # === [V29 VRAM/RAM Extreme GC] ===
    eval_venv.close()
    venv.close()
    del model
    gc.collect()
    th.cuda.empty_cache()
    
    return float(np.nan_to_num(score, nan=-1.0))

def run_v29():
    if MODE == "TUNE":
        print(f"[V29] Universal Alpha TUNE start: {STUDY_NAME}")
        study = optuna.create_study(study_name=STUDY_NAME, direction="maximize", storage=STUDY_DB, load_if_exists=True)
        study.optimize(objective, n_trials=100) 
    else:
        print(f"[V29] Train phase implementation pending")

if __name__ == "__main__":
    run_v29()
