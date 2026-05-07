import os
import glob
import json
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium import spaces
from tqdm.auto import tqdm

# 기존 유니버스 및 추출기 임포트
from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE, TCN6LayerExtractor, STAGE_3_COINS
from v26_0_env import V26_HeritageSniperEnv

warnings.filterwarnings("ignore")

# ── [V35] 설정 ──────────────────────────────────
MODEL_DIR    = "elite_weights/v35_snapshots"
LOG_DIR      = "v35_logs"
# [V35] V34의 최고 우수 모델(Score 111.7)을 베이스로 전이학습 시작
BASE_MODEL   = "elite_weights/v34_snapshots/v34_elite_rank_score111.7_step29900000"
DATA_DIR     = "data_storage"
TOTAL_TIMESTEPS = 30000000 # 3,000만 스텝 (Long/Short 양방향 완벽 정복)
LEARNING_RATE   = 5e-6     # 시니어 엔지니어의 황금 비율
CURRENT_COINS   = STAGE_3_COINS # 50개 코인 전수 대상
N_ENVS          = 16 
N_STACK         = 4          
EVAL_FREQ       = 50000
FULL_FILE_SUFFIX = "_5m_full.parquet"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── [V35] 전방위 사격 환경 (Long/Short Sniper) ───────────────
class V35_FinalEnv(V29_Universal_Env):
    def __init__(self, coin_list, data_dir, rank=0, **kwargs):
        self.data_dir = data_dir
        self.rank = rank
        self.full_universe = coin_list
        super().__init__(data_dir=data_dir, coin_files=[f"{c}{FULL_FILE_SUFFIX}" for c in coin_list], coin_id=rank, split_type=None, **kwargs)
        
        # [Surgical Edit] V35 핵심: 이산형(Discrete) 액션 공간 강제 선언
        # 0: 관망, 1: 롱(Long), 2: 숏(Short) - 이제 숏의 시대가 열린다.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        # 필터 해제 유지
        self.adx_th = 0.0
        self.far_th = 10.0
        self.max_disp = 10.0
        
        self.mfe = 0.0
        self.entry_price = 0.0

    def _load_coin_data(self, coin_file):
        path = os.path.join(self.data_dir, coin_file)
        if not os.path.exists(path): return None
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        for s in [20, 60, 200]:
            df[f"h1_ema_{s}"] = df["close"].ewm(span=s*12, adjust=False).mean()
        df["actual_h1_ema_200"] = df["h1_ema_200"].copy()
        df["h1_ema_200"] = df["close"] 
        # Live와 동일한 ATR 로직
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)
        return df.dropna().reset_index(drop=True)

    def _open_position_v35(self, side):
        row = self.df.iloc[self.current_step]
        close_now = float(row["close"])
        atr_val = float(row.get("atr_raw", 0.01)) * close_now
        sl_dist = 3.835 * atr_val
        sl_price = (close_now - sl_dist) if side == "long" else (close_now + sl_dist)
        
        self.position = {
            "dir": side,
            "entry_price": close_now,
            "sl_price": sl_price,
            "size": 1.0,
            "be_activated": False,
            "duration": 0
        }
        self.mfe = 0.0
        self.mae = 0.0
        self.entry_price = close_now
        return 0.0, False, 0.0, 0.0

    def step(self, action):
        reward = 0.0
        done = False
        info = {"coin": self.current_coin.split('_')[0] if hasattr(self, "current_coin") else "UNKNOWN"}
        
        just_opened = False
        if self.position is None:
            # action 0: 관망 / 1: 롱 / 2: 숏
            if action == 1:
                self._open_position_v35("long")
                just_opened = True
            elif action == 2:
                self._open_position_v35("short")
                just_opened = True
        
        if self.position is not None and not just_opened:
            self.position["duration"] += 1
            row = self.df.iloc[self.current_step]
            hi, lo, close_now = float(row["high"]), float(row["low"]), float(row["close"])
            
            entry_p = self.position["entry_price"]
            side = self.position["dir"]
            sl_p = self.position["sl_price"]
            
            if side == "long":
                mfe_now = (hi / entry_p - 1.0)
                mae_now = (lo / entry_p - 1.0)
                curr_pnl = (close_now / entry_p - 1.0)
            else:
                mfe_now = (1.0 - lo / entry_p)
                mae_now = (1.0 - hi / entry_p)
                curr_pnl = (1.0 - close_now / entry_p)
            
            self.mfe = max(self.mfe, mfe_now)
            self.mae = min(self.mae, mae_now)
            
            exit_triggered = False
            if (side == "long" and lo <= sl_p) or (side == "short" and hi >= sl_p):
                exit_triggered = True
                reason = "Stop_Loss"
            
            if not exit_triggered:
                if self.mfe >= 0.02 and curr_pnl < self.mfe * 0.7:
                    exit_triggered = True
                    reason = "Trailing_Stop_Mirror"

            if exit_triggered:
                final_mfe_val = float(self.mfe)
                final_mae_val = float(self.mae)
                
                r, w, m, n = self._close_position() 
                reward += r
                
                pnl_pct = (curr_pnl - 0.001) 
                regret = (final_mfe_val - pnl_pct) if final_mfe_val > pnl_pct else 0.0
                capture = (pnl_pct / final_mfe_val) if final_mfe_val > 0 else 0.0
                
                info.update({
                    "trade_closed": True,
                    "trade_Actual_PnL": pnl_pct,
                    "trade_Exit_Reason": reason,
                    "trade_MFE": final_mfe_val,
                    "trade_MAE": final_mae_val,
                    "trade_Regret": regret,
                    "trade_Capture": capture
                })
                done = True

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            if self.position:
                r, w, m, n = self._close_position()
                reward += r
                info.update({"trade_closed": True, "trade_Actual_PnL": n, "trade_Exit_Reason": "End_of_Data"})
            done = True

        atr_pct = float(self.df.iloc[self.current_step].get("atr_raw", 0.01))
        norm_reward = reward / (max(0.001, atr_pct) * 100.0)
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(norm_reward), done, False, info

def make_env(rank):
    return lambda: V35_FinalEnv(coin_list=CURRENT_COINS, data_dir=DATA_DIR, rank=rank)

def calculate_v35_score(pnl, trades, pf, wr, mdd, mae, regret, capture):
    s_pnl = pnl * 10.0 if pnl > 0 else pnl * 30.0
    s_trade = min(trades * 0.1, 10.0)
    s_pf = (pf - 1.5) * 15.0
    s_wr = (wr - 55.0) * 1.0
    p_mdd = mdd * -0.2
    p_mae = mae * -0.5
    p_reg = regret * -0.3
    b_cap = capture * 20.0
    return s_pnl + s_trade + s_pf + s_wr + p_mdd + p_mae + p_reg + b_cap

# ── [V35] 콜백 시스템 ──────────────────────────────────
class V35EliteModelCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_path = os.path.join(LOG_DIR, "v35_trades_main.csv")
        self.hunt_log = os.path.join(LOG_DIR, "v35_hunt_log.csv")
        self.best_score = -999.0
        self.trade_buffer = []
        
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=["step", "coin", "pnl", "mfe", "mae", "regret", "capture", "reason"]).to_csv(self.log_path, index=False)
        if not os.path.exists(self.hunt_log):
            pd.DataFrame(columns=["time", "step", "score", "pnl", "trades", "wr", "pf", "mdd", "mae", "reg", "cap", "snap"]).to_csv(self.hunt_log, index=False)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("trade_closed"):
                self.trade_buffer.append({
                    "step": self.num_timesteps,
                    "coin": info.get("coin", "UNKNOWN"),
                    "pnl": info.get("trade_Actual_PnL", 0.0),
                    "mfe": info.get("trade_MFE", 0.0),
                    "mae": info.get("trade_MAE", 0.0),
                    "regret": info.get("trade_Regret", 0.0),
                    "capture": info.get("trade_Capture", 0.0),
                    "reason": info.get("trade_Exit_Reason", "N/A")
                })
        
        if self.num_timesteps > 0 and self.num_timesteps % 100000 == 0:
            self._process_checkpoint()
        return True

    def _process_checkpoint(self):
        if not self.trade_buffer: return
        df = pd.DataFrame(self.trade_buffer)
        df.to_csv(self.log_path, mode='a', header=False, index=False)
        
        trades = len(df)
        wr = (df['pnl'] > 0).mean() * 100
        pnl_sum = df['pnl'].sum() * 100
        pos_sum = df[df['pnl'] > 0]['pnl'].sum()
        neg_sum = abs(df[df['pnl'] < 0]['pnl'].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else (5.0 if pos_sum > 0 else 0)
        mdd = df['pnl'].expanding().sum().min() * -100
        mae = df['mae'].mean() * 100
        reg = df['regret'].mean() * 100
        cap = df['capture'].mean() * 100
        
        score = calculate_v35_score(pnl_sum/trades, trades, pf, wr, mdd, mae, reg, cap/100.0)
        
        if not hasattr(self, 'top_scores'): self.top_scores = []
        
        snap_status = ""
        if len(self.top_scores) < 5 or score > min([s for s, p in self.top_scores]):
            save_name = f"v35_elite_rank_score{score:.1f}_step{self.num_timesteps}"
            save_path = os.path.join(MODEL_DIR, save_name)
            self.model.save(save_path)
            self.top_scores.append((score, save_path))
            self.top_scores.sort(key=lambda x: x[0], reverse=True)
            self.top_scores = self.top_scores[:5] 
            snap_status = "SAVED"
            if score > self.best_score: self.best_score = score
        
        new_hunt = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "step": self.num_timesteps,
            "score": score, "pnl": pnl_sum, "trades": trades, "wr": wr, "pf": pf,
            "mdd": mdd, "mae": mae, "reg": reg, "cap": cap, "snap": snap_status
        }
        pd.DataFrame([new_hunt]).to_csv(self.hunt_log, mode='a', header=False, index=False)
        self.trade_buffer = [] 

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="V35 Training (L/S)")
    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True
    def _on_training_end(self):
        self.pbar.close()

if __name__ == "__main__":
    print(f"[*] Starting V35 Long/Short Sniper Training (Discrete Action Space)")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = VecFrameStack(SubprocVecEnv([make_env(i) for i in range(N_ENVS)]), n_stack=N_STACK)
    
    policy_kwargs = dict(
        features_extractor_class=TCN6LayerExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[256, 128])
    )
    
    # [V35] PPO 모델 생성 - Discrete(3) 액션 공간은 여기서 자동 인식됨
    model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, device="cuda", learning_rate=LEARNING_RATE)
    
    # [V35] V34의 우수한 특징 추출기(Backbone) 지식 계승
    if os.path.exists(BASE_MODEL):
        print(f"[*] Loading knowledge from V34 Top Model: {BASE_MODEL}")
        _, params, _ = load_from_zip_file(BASE_MODEL, device="cuda")
        
        # [Surgical Fix] Action Head(action_net)의 shape 불일치(1 vs 3) 해결
        # 특징 추출기(Backbone) 등 나머지 가중치만 필터링하여 로드합니다.
        new_state_dict = {k: v for k, v in params["policy"].items() if "action_net" not in k}
        
        # strict=False를 통해 구조가 달라진 action_head는 제외하고 특징 추출기만 로드
        model.policy.load_state_dict(new_state_dict, strict=False)
        print(f"[*] Backbone knowledge transferred successfully. Action head re-initialized for Discrete(3).")
    
    model.save(os.path.join(MODEL_DIR, "v35_baseline_v34_knowledge"))
    
    eval_env = VecFrameStack(DummyVecEnv([make_env(0)]), n_stack=N_STACK)
    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=EVAL_FREQ, deterministic=True)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, V35EliteModelCallback(), ProgressBarCallback(TOTAL_TIMESTEPS)])
    model.save(os.path.join(MODEL_DIR, "v35_final_production_model"))
