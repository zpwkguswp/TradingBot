"""
V33 Full-History Universal Training Engine (V30/V29 Architecture)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 데이터  : data_storage/{COIN}_5m_full.parquet (RAW OHLCV)
■ 피처    : V29 Universal Alpha 37-Dim (On-the-fly Engineering)
■ 보상    : ATR-Normalized Reward (Volatility Adjusted)
■ 모델    : TCN6LayerExtractor + PPO (백지 초기화)
■ 환경    : V29_Universal_Env 상속 (V30 커리큘럼 하이퍼파라미터 적용)
"""

import gc
import os
import random
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")

from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE, TCN6LayerExtractor

# ──────────────────────────────────────────────────────────────────
#  경로 / 상수
# ──────────────────────────────────────────────────────────────────
DATA_DIR  = "data_storage"
ELITE_DIR = "elite_weights"
LOG_DIR   = "v33_logs"
os.makedirs(ELITE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

DEVICE  = "cuda" if th.cuda.is_available() else "cpu"
N_STACK = 4
FULL_FILE_SUFFIX = "_5m_full.parquet"
BEST_MODEL_PATH = os.path.join(ELITE_DIR, "v33_best_model.zip")

# ──────────────────────────────────────────────────────────────────
#  V33 환경 설정: V30 Stage 3와 유사하게 필터 대폭 완화
# ──────────────────────────────────────────────────────────────────
TRAIN_ENV_PARAMS = dict(
    target_profit = 0.010,
    far_th        = 5.0,    # 이격도 제한 거의 해제
    sl_atr_coef   = 3.0,
    adx_th        = 0.0,    # 횡보장에서도 진입 허용 (V30 룰)
    max_disp      = 1.0,    # 정수리 필터 해제
    trail_act     = 0.015,
)

EVAL_ENV_PARAMS = dict(
    target_profit = 0.008,
    far_th        = 2.204,
    sl_atr_coef   = 3.835,
    adx_th        = 0.169,
    max_disp      = 0.058,
    trail_act     = 0.020,
)

# ══════════════════════════════════════════════════════════════════
#  1. V33_FullHistoryEnv: RAW 데이터를 V29 호환 피처로 변환
# ══════════════════════════════════════════════════════════════════
class V33_FullHistoryEnv(V29_Universal_Env):
    def __init__(self, coin_list, data_dir, split="train", **kwargs):
        self.data_dir = data_dir
        self.split = split
        self.full_universe = coin_list
        
        # 임시 초기화
        super().__init__(
            data_dir=data_dir,
            coin_files=[f"BTCUSDT{FULL_FILE_SUFFIX}"],
            coin_id=0,
            split_type=None,
            **kwargs
        )
        
        self.cached_dfs = {}
        self.valid_coins = []
        
        print(f"[V33 Env] Generating V29/V30 features for {len(coin_list)} coins...")
        
        for ticker in coin_list:
            filename = f"{ticker}{FULL_FILE_SUFFIX}"
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path): continue
            
            try:
                df = pd.read_parquet(path).reset_index(drop=True)
                
                # ─── V29/V30 호환 피처 엔지니어링 ───
                # 1. EMA
                for s in [20, 60, 200]:
                    df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
                    df[f"h1_ema_{s}"] = df["close"].ewm(span=s*12, adjust=False).mean() # 5m * 12 = 1h

                # 2. ATR & Volatility
                hl = df["high"] - df["low"]
                hc = (df["high"] - df["close"].shift()).abs()
                lc = (df["low"] - df["close"].shift()).abs()
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                df["atr_raw"] = tr.rolling(14).mean() / df["close"]
                df["h1_atr_abs"] = tr.rolling(14*12).mean()

                # 3. ADX & Macro
                up = df["high"] - df["high"].shift()
                dn = df["low"].shift() - df["low"]
                pdm = np.where((up > dn) & (up > 0), up, 0)
                mdm = np.where((dn > up) & (dn > 0), dn, 0)
                pdi = 100 * (pd.Series(pdm).rolling(14).mean() / tr.rolling(14).mean())
                mdi = 100 * (pd.Series(mdm).rolling(14).mean() / tr.rolling(14).mean())
                dx = 100 * (abs(pdi - mdi) / (pdi + mdi + 1e-9))
                df["adx_14"] = dx.rolling(14).mean()
                df["macro_adx"] = dx.rolling(14*12).mean() / 100.0
                df["macro_atr_ratio"] = df["atr_raw"] / (df["atr_raw"].rolling(14*12).mean() + 1e-9)
                df["macro_bb_width"] = (df["high"].rolling(14*12).max() - df["low"].rolling(14*12).min()) / df["close"]

                # 4. Disparity
                df["disparity_200_atr"] = (df["close"] - df["ema_200"]) / (tr.rolling(200).mean() + 1e-9)
                df["disparity_60_atr"] = (df["close"] - df["ema_60"]) / (tr.rolling(60).mean() + 1e-9)
                df["v26_disparity"] = ((df["close"] - df["ema_60"]) / df["ema_60"]) * 100.0
                df["delta_thresh"] = df["atr_raw"] * 0.5

                # 5. TCN Dummy (RSI rank based momentum)
                rel_str = df["close"].pct_change(5).rolling(10).rank(pct=True).fillna(0.5)
                df["tcn_p_up"]   = rel_str
                df["tcn_p_down"] = 1.0 - rel_str
                df["tcn_p_chop"] = 0.33
                df["tcn_p_var"]  = 0.01

                df = df.dropna().reset_index(drop=True)
                
                # Split 90/10
                split_idx = int(len(df) * 0.9)
                if self.split == "train":
                    df = df.iloc[:split_idx].reset_index(drop=True)
                else:
                    df = df.iloc[split_idx:].reset_index(drop=True)
                
                if len(df) > 1000:
                    self.cached_dfs[filename] = df
                    self.valid_coins.append(filename)
            except Exception as e:
                print(f"Error engineering {ticker}: {e}")
                continue
                
        if not self.valid_coins:
            raise RuntimeError("No valid data found.")
            
        self._max_ep_steps = 10000 
        self._ep_step_cnt = 0

    def reset(self, seed=None, options=None):
        filename = random.choice(self.valid_coins)
        ticker = filename.replace(FULL_FILE_SUFFIX, "")
        self.coin_id = self.full_universe.index(ticker) if ticker in self.full_universe else 0
        
        self.coin_files = [filename]
        self.current_coin = filename
        self.df = self.cached_dfs[filename]
        
        self._ep_step_cnt = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # 5분봉 특성상 에피소드가 너무 길면 학습 효율이 떨어지므로 제한
        obs, reward, done, truncated, info = super().step(action)
        self._ep_step_cnt += 1
        if self._ep_step_cnt >= self._max_ep_steps:
            done = True
        return obs, reward, done, truncated, info

# ══════════════════════════════════════════════════════════════════
#  2. 훈련 설정 및 팩토리
# ══════════════════════════════════════════════════════════════════
def make_env_fn(coin_list, split, params):
    def _init():
        return V33_FullHistoryEnv(coin_list=coin_list, data_dir=DATA_DIR, split=split, **params)
    return _init

def build_venv(coin_list, split, n_envs, params):
    fns = [make_env_fn(coin_list, split, params) for _ in range(n_envs)]
    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        venv = SubprocVecEnv(fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv(fns)
    from stable_baselines3.common.vec_env import VecFrameStack
    return VecFrameStack(venv, n_stack=N_STACK)

# ══════════════════════════════════════════════════════════════════
#  3. 커스텀 콜백: 거래 로깅 및 성과 추적
# ══════════════════════════════════════════════════════════════════
class V33TrainCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.trade_log_path = os.path.join(LOG_DIR, "v33_trades.csv")
        self._trade_buf = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("trade_closed"):
                self._trade_buf.append({
                    "step":   self.num_timesteps,
                    "pnl":     info.get("trade_Actual_PnL", 0.0),
                    "reason":  info.get("trade_Exit_Reason", ""),
                    "coin":    info.get("coin_id", -1),
                    "equity":  info.get("bot_equity", 1.0),
                    "mdd":     info.get("bot_mdd", 0.0)
                })
        
        # 즉시 기록하여 사용자가 상황판에서 바로 볼 수 있게 함
        if self._trade_buf:
            self._flush_logs()
        return True

class BestModelCopyCallback(BaseCallback):
    """최고 모델 갱신 시 v33_best_model.zip으로 즉시 복사"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_best_mtime = 0

    def _on_step(self) -> bool:
        src = os.path.join(ELITE_DIR, "best_model.zip")
        if os.path.exists(src):
            mtime = os.path.getmtime(src)
            if mtime > self.last_best_mtime:
                import shutil
                shutil.copy(src, BEST_MODEL_PATH)
                self.last_best_mtime = mtime
                if self.verbose > 0:
                    print(f"\n[Callback] New best model copied to {BEST_MODEL_PATH}")
        return True

    def _flush_logs(self):
        if not self._trade_buf: return
        df = pd.DataFrame(self._trade_buf)
        hdr = not os.path.exists(self.trade_log_path)
        df.to_csv(self.trade_log_path, mode="a", index=False, header=hdr)
        self._trade_buf = []

    def _on_training_end(self):
        self._flush_logs()

# ══════════════════════════════════════════════════════════════════
#  4. 메인 훈련 함수
# ══════════════════════════════════════════════════════════════════
def run_v33_training():
    print("\n" + "=" * 60)
    print("  V33 Universal 5m Training Engine Start")
    print("=" * 60)
    print(f"  Device   : {DEVICE}")
    print(f"  Universe : {len(FULL_UNIVERSE)} coins")
    print(f"  Split    : 90% Train / 10% Eval (Chronological)")
    print("=" * 60 + "\n")

    # 1. 환경 생성
    n_train_envs = 4
    train_venv = build_venv(FULL_UNIVERSE, "train", n_train_envs, TRAIN_ENV_PARAMS)
    eval_venv = build_venv(FULL_UNIVERSE, "eval", 1, EVAL_ENV_PARAMS)

    # 2. PPO 모델 초기화
    policy_kwargs = dict(
        features_extractor_class=TCN6LayerExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[256, 128]),
    )

    model = PPO(
        "MlpPolicy",
        train_venv,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-5,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device=DEVICE
    )

    # 3. 콜백 설정
    eval_callback = EvalCallback(
        eval_env=eval_venv,
        best_model_save_path=ELITE_DIR,
        log_path=LOG_DIR,
        eval_freq=100000 // n_train_envs,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1
    )
    v33_callback = V33TrainCallback()
    copy_callback = BestModelCopyCallback()

    # 4. 훈련 개시
    print("\n[Training] 5,000,000 steps sequence started...")
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[eval_callback, v33_callback, copy_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted.")

    # 5. 최종 저장
    model.save(os.path.join(ELITE_DIR, "v33_final_model"))
    
    src = os.path.join(ELITE_DIR, "best_model.zip")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, BEST_MODEL_PATH)
        print(f"\n[Success] Best model saved to {BEST_MODEL_PATH}")
    
    train_venv.close()
    eval_venv.close()

if __name__ == "__main__":
    run_v33_training()
