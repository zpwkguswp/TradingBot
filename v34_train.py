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

# ── [V34] 설정 ──────────────────────────────────
MODEL_DIR    = "elite_weights/v34_snapshots"
LOG_DIR      = "v34_logs"
BASE_MODEL   = "elite_weights/v33_3_snapshots/stage3_run018_score31.3_20260503_180454/model.zip" # Run 18 베이스
DATA_DIR     = "data_storage"
TOTAL_TIMESTEPS = 30000000 # 3,000만 스텝 (Efficiency Optimization)
LEARNING_RATE   = 5e-6     # 기존 룰 준수
CURRENT_COINS   = STAGE_3_COINS # 50개 코인 전수 대상
N_ENVS          = 16 
N_STACK         = 4          
EVAL_FREQ       = 50000
FULL_FILE_SUFFIX = "_5m_full.parquet"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── [V34] 실전 동기화 환경 (Mirror Env) ───────────────
class V34_FinalEnv(V29_Universal_Env):
    def __init__(self, coin_list, data_dir, rank=0, **kwargs):
        self.data_dir = data_dir
        self.rank = rank
        self.full_universe = coin_list
        super().__init__(data_dir=data_dir, coin_files=[f"{c}{FULL_FILE_SUFFIX}" for c in coin_list], coin_id=rank, split_type=None, **kwargs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
        
        # [V34] 하드코딩 필터 강제 해제 (모델이 직접 학습하도록 유도)
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
        # [Surgical Edit 1] Live와 100% 동기화: True Range -> 단순 H-L 방식으로 교체
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)
        return df.dropna().reset_index(drop=True)

    def _open_position_v34(self, side):
        row = self.df.iloc[self.current_step]
        close_now = float(row["close"])
        # ATR 기반 손절가 설정 (v28/v29 규격 준수)
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
        self.mae = 0.0 # [V34] MAE 추적 추가
        self.entry_price = close_now
        return 0.0, False, 0.0, 0.0

    def step(self, action):
        reward = 0.0
        done = False
        info = {"coin": self.current_coin.split('_')[0] if hasattr(self, "current_coin") else "UNKNOWN"}
        
        # 1. 진입 로직
        # [Surgical Edit 2] just_opened 플래그 도입: 진입 직후 동일 봉 평가 차단
        just_opened = False
        if self.position is None:
            if action == 1:
                self._open_position_v34("long")
                just_opened = True
            elif action == 2:
                self._open_position_v34("short")
                just_opened = True
        
        # 2. 청산 및 상태 업데이트 (방금 열린 포지션은 다음 캔들부터 감시)
        if self.position is not None and not just_opened:
            self.position["duration"] += 1
            row = self.df.iloc[self.current_step]
            hi, lo, close_now = float(row["high"]), float(row["low"]), float(row["close"])
            
            entry_p = self.position["entry_price"]
            side = self.position["dir"]
            sl_p = self.position["sl_price"]
            
            # [V34 Precision Metrics] 봉 내부의 High/Low를 모두 반영하여 진짜 MAE/MFE 추적
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
                # 실전 트레일링 스탑 (MFE 기반)
                if self.mfe >= 0.02 and curr_pnl < self.mfe * 0.7:
                    exit_triggered = True
                    reason = "Trailing_Stop_Mirror"

            if exit_triggered:
                # [V34 Fix] 조상님이 초기화하기 전에 미리 지표를 안전하게 복사
                final_mfe_val = float(self.mfe)
                final_mae_val = float(self.mae)
                
                r, w, m, n = self._close_position() # 여기서 조상님이 self.mfe=0 으로 리셋함
                reward += r
                
                # [V34] 백분율 기반 정밀 지표 산출
                pnl_pct = (curr_pnl - 0.001) # 수수료/슬리피지 약 0.1% 반영
                regret = (final_mfe_val - pnl_pct) if final_mfe_val > pnl_pct else 0.0
                capture = (pnl_pct / final_mfe_val) if final_mfe_val > 0 else 0.0
                
                info.update({
                    "trade_closed": True,
                    "trade_Actual_PnL": pnl_pct, # 백분율로 통일
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
    return lambda: V34_FinalEnv(coin_list=CURRENT_COINS, data_dir=DATA_DIR, rank=rank)

def calculate_v34_score(pnl, trades, pf, wr, mdd, mae, regret, capture):
    s_pnl = pnl * 10.0 if pnl > 0 else pnl * 30.0
    s_trade = min(trades * 0.1, 10.0)
    s_pf = (pf - 1.5) * 15.0
    s_wr = (wr - 55.0) * 1.0
    p_mdd = mdd * -0.2
    p_mae = mae * -0.5
    p_reg = regret * -0.3
    b_cap = capture * 20.0
    return s_pnl + s_trade + s_pf + s_wr + p_mdd + p_mae + p_reg + b_cap

# ── [V34] 콜백 시스템 ──────────────────────────────────
class V34EliteModelCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_path = os.path.join(LOG_DIR, "v34_trades_main.csv")
        self.hunt_log = os.path.join(LOG_DIR, "v34_hunt_log.csv")
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
        
        score = calculate_v34_score(pnl_sum/trades, trades, pf, wr, mdd, mae, reg, cap/100.0)
        
        # [V34 Elite Hunter v2] Top-5 모델 관리
        if not hasattr(self, 'top_scores'): self.top_scores = []
        
        snap_status = ""
        # 현재 점수가 Top-5 진입 대상인지 확인
        if len(self.top_scores) < 5 or score > min([s for s, p in self.top_scores]):
            save_name = f"v34_elite_rank_score{score:.1f}_step{self.num_timesteps}"
            save_path = os.path.join(MODEL_DIR, save_name)
            self.model.save(save_path)
            self.top_scores.append((score, save_path))
            self.top_scores.sort(key=lambda x: x[0], reverse=True)
            self.top_scores = self.top_scores[:5] # 상위 5개 유지
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
        self.pbar = tqdm(total=self.total_timesteps, desc="V34 Training")
    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True
    def _on_training_end(self):
        self.pbar.close()

if __name__ == "__main__":
    print(f"[*] Starting V34 Grand Finale Training (100% Data + Trailing Stop Sync)")
    
    # [V34 자산 보존] 기존 기록을 삭제하지 않고 유지하며 새 훈련 시작
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = VecFrameStack(SubprocVecEnv([make_env(i) for i in range(N_ENVS)]), n_stack=N_STACK)
    
    policy_kwargs = dict(
        features_extractor_class=TCN6LayerExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[256, 128])
    )
    
    model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, device="cuda", learning_rate=LEARNING_RATE)
    _, params, _ = load_from_zip_file(BASE_MODEL, device="cuda")
    model.policy.load_state_dict(params["policy"], strict=False)
    
    # [V34] 베이스라인 모델 즉시 저장 (기준점 확보)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "v34_baseline_run018"))
    
    # [V34 Fix] eval_env의 rank를 100 -> 0으로 변경하여 임베딩 인덱스 초과(CUDA 에러) 방지
    eval_env = VecFrameStack(DummyVecEnv([make_env(0)]), n_stack=N_STACK)
    
    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=EVAL_FREQ, deterministic=True)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, V34EliteModelCallback(), ProgressBarCallback(TOTAL_TIMESTEPS)])
    model.save(os.path.join(MODEL_DIR, "v34_final_production_model"))
