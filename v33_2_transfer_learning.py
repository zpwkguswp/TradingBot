import os
import glob
import json
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime

# Stable Baselines3 관련
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium import spaces

# 기존 V30/V29 파일 임포트
from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE, TCN6LayerExtractor, STAGE_1_COINS, STAGE_2_COINS, STAGE_3_COINS

# 필터 우회용 최상위 클래스
from v26_0_env import V26_HeritageSniperEnv
from v26_3_env import V26_3_HeritageDisparityEnv

warnings.filterwarnings("ignore")

# [V33-2] 설정 로드 함수
def load_config():
    config_path = "v33_2_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "learning_rate": 5e-6, "regret_penalty_weight": 4.0, "mae_penalty_threshold": -0.005,
        "capture_bonus_weight": 0.2, "total_timesteps": 300000, "run_idx": 1, "cum_start_step": 0
    }

# ── 설정 ──────────────────────────────────
DATA_DIR     = "data_storage"
CONFIG = load_config()
LOG_DIR = CONFIG.get("log_dir", "v33_2_logs")
MODEL_DIR    = "elite_weights"
BASE_MODEL   = "elite_weights/v30_best_model_2h.zip" 
RESUME_MODEL = "elite_weights/v33_2_final_model.zip" # 마지막 저장된 모델
BEST_MODEL   = "elite_weights/best_model.zip"      # EvalCallback이 저장한 최고 모델
FINAL_MODEL  = "v33_2_final_model"

# CONFIG는 상단에서 이미 로드됨

# 전역 상수 설정
STAGE = CONFIG.get("stage", 1)
TOTAL_TIMESTEPS = CONFIG.get("total_timesteps", 1000000)
LEARNING_RATE = CONFIG.get("learning_rate", 5e-6)
REGRET_W = CONFIG.get("regret_penalty_weight", 4.0)
MAE_TH = CONFIG.get("mae_penalty_threshold", -0.005)
CAP_W = CONFIG.get("capture_bonus_weight", 0.2)
TARGET_PF = CONFIG.get("target_pf", 1.5)

# 스테이지별 코인 리스트 결정
if STAGE == 1:
    CURRENT_COINS = STAGE_1_COINS
elif STAGE == 2:
    CURRENT_COINS = STAGE_2_COINS
else:
    CURRENT_COINS = STAGE_3_COINS

print(f"[*] Current Stage: {STAGE} | Coins: {len(CURRENT_COINS)}")

EVAL_FREQ       = 50_000 
N_ENVS          = 16 
N_STACK         = 4          
N_EVAL_ENVS     = 1          
N_EVAL_EPISODES = 5          

FULL_FILE_SUFFIX = "_5m_full.parquet"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

from tqdm.auto import tqdm

# ── [V33-2] 콜백 섹션 ──────────────────────────────────
class TradeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_path = os.path.join(LOG_DIR, "v33_2_trades_main.csv")
    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            pnl = info.get("trade_Actual_PnL") or info.get("pnl")
            if pnl is not None:
                try:
                    # PnL 포착력 강화 (여러 경로 확인)
                    pnl = info.get("trade_Actual_PnL", 0.0)
                    if pnl == 0.0: pnl = info.get("pnl", 0.0)
                    
                    trade_data = {
                        "run": CONFIG.get("run_idx", 1),
                        "stage": CONFIG.get("stage", 1),
                        "train_step": self.num_timesteps + CONFIG.get("cum_start_step", 0),
                        "coin": info.get("coin_id", 0),
                        "pnl": float(pnl),
                        "mfe": float(info.get("trade_MFE", 0.0)),
                        "mae": float(info.get("trade_MAE", 0.0)),
                        "regret": float(info.get("trade_Regret", 0.0)),
                        "capture": float(info.get("trade_Capture", 0.0)),
                        "entry_disp": float(info.get("trade_Entry_Disp", 0.0)),
                        "is_counter": int(info.get("trade_IsCounter", 0)),
                        "reason": str(info.get("trade_Exit_Reason", "Closed")),
                        "score": float(info.get("last_reward", 0.0))
                    }
                    pd.DataFrame([trade_data]).to_csv(self.log_path, mode='a', header=not os.path.exists(self.log_path), index=False)
                except: pass
        return True

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_phase = 0
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        # Phase 결정
        phase = 1 if current_step < 2_000_000 else (2 if current_step < 4_000_000 else 3)
        
    def _on_step(self) -> bool:
        # Stage 1 정밀 튜닝 모드: 커리큘럼 전환을 비활성화하고 Stage 1(BTC/ETH)에 고정
        return True

class Top5ModelCallback(BaseCallback):
    def __init__(self, eval_cb, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_cb = eval_cb # EvalCallback 참조를 직접 받음
        self.save_path = save_path
        self.top_models = [] # List of (score, filename)
        self.last_best_score = -np.inf
        
    def _on_step(self) -> bool:
        # EvalCallback이 저장한 best_model.zip을 감시하여 새로운 최고점 모델을 복사함
        best_model_path = os.path.join(self.save_path, "best_model.zip")
        if os.path.exists(best_model_path):
            # EvalCallback에서 최근 최고점을 가져옴
            current_best_score = self.eval_cb.best_mean_reward
            
            # 아직 평가가 이루어지지 않았거나, 이전 최고점과 같다면 건너뜀
            if current_best_score == -np.inf or current_best_score <= self.last_best_score:
                return True
                
            self.last_best_score = current_best_score
            filename = f"v33_2_best_score_{current_best_score:.2f}.zip"
            dest_path = os.path.join(self.save_path, filename)
            
            # 파일 복사
            import shutil
            shutil.copy(best_model_path, dest_path)
            self.top_models.append((current_best_score, filename))
            self.top_models.sort(key=lambda x: x[0], reverse=True)
            
            # 상위 5개 초과분 삭제
            if len(self.top_models) > 5:
                _, old_file = self.top_models.pop()
                old_path = os.path.join(self.save_path, old_file)
                if os.path.exists(old_path): os.remove(old_path)
                
            if self.verbose > 0:
                print(f"\n[Top5] New best model saved: {filename} (Total: {len(self.top_models)})")
        return True

class PFTargetStopCallback(BaseCallback):
    def __init__(self, target_pf, min_trades=100, verbose=1):
        super().__init__(verbose)
        self.target_pf = target_pf
        self.min_trades = min_trades
        self.recent_pnls = []
        self.reached = False

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            pnl = info.get("trade_Actual_PnL") or info.get("pnl")
            if pnl is not None:
                self.recent_pnls.append(float(pnl))
                if len(self.recent_pnls) > 500: self.recent_pnls.pop(0)

        if len(self.recent_pnls) >= self.min_trades and not self.reached:
            pos_sum = sum([p for p in self.recent_pnls if p > 0])
            neg_sum = abs(sum([p for p in self.recent_pnls if p < 0]))
            current_pf = pos_sum / neg_sum if neg_sum > 0 else 10.0
            
            if current_pf >= self.target_pf:
                print(f"\n[!] Target PF {self.target_pf} Reached! (Current: {current_pf:.2f})")
                print(f"[*] Continuing training to reach 1M steps (Fixed Stage Learning)...")
                self.reached = True
                # return False # 조기 종료 비활성화
        return True

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None; self.total_timesteps = total_timesteps
    def _on_training_start(self): self.pbar = tqdm(total=self.total_timesteps, initial=self.num_timesteps, desc="Training Progress")
    def _on_step(self) -> bool:
        if self.pbar: self.pbar.update(self.training_env.num_envs)
        return True
    def _on_training_end(self):
        if self.pbar: self.pbar.close()

# ── [V33-2] 환경 섹션 ──────────────────
class V33_FullHistoryEnv(V29_Universal_Env):
    def __init__(self, coin_list, data_dir, rank=0, **kwargs):
        self.data_dir = data_dir
        self.rank = rank
        self.full_universe = coin_list
        self._ep_step_cnt = 0; self.trade_history = []; self.history_len = 50; self._max_ep_steps = 5000 
        
        # MFE/MAE 및 수익률 트래킹 초기화
        self.mfe = 0.0; self.mae = 0.0; self.entry_price = 0.0; self.last_net_pnl = 0.0; self.is_counter = False
        
        kwargs.pop("obs_dim", None)
        kwargs.pop("adx_th", None); kwargs.pop("max_disp", None); kwargs.pop("hold_signal_thr", None)
        
        # [Fix] 파일이 실제로 존재하는 코인만 필터링 (RNDR 등 누락 방지)
        existing_coins = []
        for c in CURRENT_COINS:
            if os.path.exists(os.path.join(data_dir, f"{c}{FULL_FILE_SUFFIX}")):
                existing_coins.append(c)
            else:
                print(f"[Warning] Data file for {c} not found. Skipping...")
        
        super().__init__(data_dir=data_dir, coin_files=[f"{c}{FULL_FILE_SUFFIX}" for c in existing_coins], coin_id=rank, split_type=None, **kwargs)
        
        # [V33-2] 초기 코인들(BTC, ETH)에 대해 MTF 지표 로드 보장
        for cf in self.coin_files:
            self._load_coin_data(cf)
            
        self.obs_dim = 37
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)
        self.env_log_path = os.path.join(LOG_DIR, f"v33_2_trades_direct_{self.rank}.csv")

    def update_universe(self, new_list):
        self.full_universe = new_list
        self.coin_files = [f"{c}{FULL_FILE_SUFFIX}" for c in new_list]

    def _check_entry_condition(self, signal_dir):
        return True 

    def _load_coin_data(self, coin_file):
        # 이미 로드되어 있고 필요한 지표가 있다면 재사용
        if coin_file in self.cached_dfs and "h1_ema_200" in self.cached_dfs[coin_file].columns:
            return self.cached_dfs[coin_file]
            
        path = os.path.join(self.data_dir, coin_file)
        if not os.path.exists(path): return None
        try:
            df = pd.read_parquet(path).reset_index(drop=True)
            for s in [20, 60, 200]:
                df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
                df[f"h1_ema_{s}"] = df["close"].ewm(span=s*12, adjust=False).mean()
            
            # [V33-2] 자율 학습을 위한 부모 필터 무력화 (진입 자유 보장)
            df["actual_h1_ema_200"] = df["h1_ema_200"].copy() # [Fix] 분석용 실제 데이터 보존
            df["h1_ema_200"] = df["close"] # [Fix] 부모 필터용 (진입 허용용)
            
            df["structural_rev_long"] = 1.0; df["structural_rev_short"] = 1.0
            df["ema_squeeze"] = 0.0; df["adx_14"] = 30.0
            
            hl, hc, lc = df["high"]-df["low"], (df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr_raw"] = tr.rolling(14).mean() / (df["close"] + 1e-9)
            
            df = df.dropna().reset_index(drop=True)
            self.cached_dfs[coin_file] = df
            return df
        except Exception as e:
            print(f"[Error] Failed to load {coin_file}: {e}")
            return None

    def _close_position(self):
        # 부모 클래스의 청산 로직을 호출하여 실제 수익률(net_pnl)을 가져옴
        reward, won, penalty, net_pnl = V26_HeritageSniperEnv._close_position(self)
        self.last_net_pnl = net_pnl
        return reward, won, penalty, net_pnl

    # [해결] 리셋 시 코인 데이터를 즉시 로드하도록 보강
    def reset(self, seed=None, options=None):
        self._ep_step_cnt = 0; self.mfe = 0.0; self.mae = 0.0; self.entry_price = 0.0; self.last_net_pnl = 0.0
        
        # [V33-2 Fix] 부모 클래스의 reset()이 로드되지 않은 코인을 선택하여 KeyError가 발생하는 것을 방지
        # 1. 먼저 우리가 코인을 하나 선택하고 로드를 보장함
        target_coin = np.random.choice(self.coin_files)
        if target_coin not in self.cached_dfs or "h1_ema_200" not in self.cached_dfs[target_coin].columns:
            res = self._load_coin_data(target_coin)
            if res is None: # 로드 실패 시 기존에 있는 코인 중 하나로 폴백
                target_coin = list(self.cached_dfs.keys())[0]
        
        # 2. 부모 클래스의 reset()이 반드시 이 target_coin을 선택하도록 잠시 coin_files를 조작
        orig_files = self.coin_files
        self.coin_files = [target_coin]
        try:
            # v24_env 등의 부모 클래스 reset() 호출
            obs, info = super().reset(seed=seed, options=options)
        finally:
            # 3. 원래의 coin_files 리스트 복구
            self.coin_files = orig_files
            
        return obs, info

    def _get_obs(self) -> np.ndarray:
        obs_33 = V26_3_HeritageDisparityEnv._get_obs(self)
        padding = np.zeros(4, dtype=np.float32)
        return np.concatenate([obs_33, padding])

    def step(self, action):
        if getattr(self, "position", None) is not None:
            self.position["duration"] = self.position.get("duration", 0) + 1
            if self.position["duration"] < 10:
                action = np.array([0.5], dtype=np.float32) 

        was_in_pos = getattr(self, "position", None) is not None
        
        # 진입 전 트래커 리셋 (새 거래 시작 시)
        if not was_in_pos:
            self.mfe = 0.0; self.mae = 0.0; self.entry_price = 0.0; self.is_counter = False

        # [V33-2] 필터 완전 개방: 부모 클래스의 추세 필터를 우회하기 위해 mock_action 조작
        # V26_3 등에서 거는 EMA-200 필터를 무력화하고 AI의 의도대로 진입함
        obs, reward, done, truncated, info = V26_HeritageSniperEnv.step(self, action)
        self._ep_step_cnt += 1
        
        is_in_pos = getattr(self, "position", None) is not None
        row = self.df.iloc[self.current_step]
        close_now = float(row["close"])
        h1_ema200 = float(row.get("actual_h1_ema_200", close_now)) # [Fix] 실제 EMA 사용
        entry_disp = abs(close_now - h1_ema200) / (h1_ema200 + 1e-9)
        
        # ── [V33-2] 자율 학습형 보상 및 역추세 관리 ──────────────────
        
        # 1. 진입 성격 판별 (보상 없음, 판별만 수행)
        if not was_in_pos and is_in_pos:
            self.entry_price = close_now
            self.entry_disparity = entry_disp # 분석을 위해 저장
            side = self.position["dir"]
            
            # 역추세 판별: 롱인데 가격 < EMA200 이거나, 숏인데 가격 > EMA200 인 경우 (Mean Reversion)
            self.is_counter = (side == "long" and close_now < h1_ema200) or (side == "short" and close_now > h1_ema200)

        # 2. 보유 중 리스크 및 탈출 관리
        if is_in_pos:
            side = self.position["dir"]
            pnl_now = (close_now / self.entry_price - 1.0) if side == "long" else (1.0 - close_now / self.entry_price)
            self.mfe = max(self.mfe, pnl_now)
            self.mae = min(self.mae, pnl_now)
            
            # [역추세 전용 Smart Escape] 역추세 매매인데 반대로 1.0% 이상 밀리면 즉시 탈출 (추세 연장 위험)
            if self.is_counter and pnl_now < -0.01:
                reward -= 0.3 # 강한 탈출 페널티
                self.position = None
                is_in_pos = False
                info["trade_Exit_Reason"] = "CounterTrend_Stop"
        # 3. 종료 시점 최종 평가 (Dynamic Alpha: 이격도 비례 보상)
        if was_in_pos and not is_in_pos:
            actual_pnl = getattr(self, "last_net_pnl", 0.0)
            
            # 지표 주입 (로그용으로 기록만 남김)
            info["trade_Actual_PnL"] = actual_pnl 
            info["trade_IsCounter"] = self.is_counter
            info["trade_Entry_Disp"] = getattr(self, "entry_disparity", 0.0) 
            info["trade_Exit_Reason"] = info.get("trade_Exit_Reason", "Closed")

            # 🌟 [V33-3 핵심] 실제 수익률을 기본 점수로 세팅 (1% 수익 = 1점)
            base_score = actual_pnl * 100.0  
            
            if actual_pnl > 0:
                # ✅ [수익을 냈을 때] 진입 성격에 따른 보상 뻥튀기 (Multiplier)
                if self.is_counter:
                    # 🎯 역추세 성공: 이격도가 클수록(위험할수록) 엄청난 추가 보상!
                    multiplier = 1.0 + (self.entry_disparity * 100.0)
                    reward += (base_score * multiplier)
                else:
                    # 🎯 추세(눌림목) 성공: 이격도가 작을수록 정상 보상, 이격도가 큰데 추세로 따라가면 보상 삭감
                    multiplier = max(0.1, 1.0 - (self.entry_disparity * 30.0))
                    reward += (base_score * multiplier)
            else:
                # ❌ [손실을 냈을 때]
                if not self.is_counter and self.entry_disparity > 0.02:
                    # 🎯 멍청한 매매 처벌: 이격도가 2% 이상 벌어진 '정수리/바닥'에서 추세 매매(추격)를 하다가 물리면 2배 가중 처벌!
                    reward += (base_score * 2.0) # base_score가 음수이므로 감점이 2배가 됨
                else:
                    # 일반적인 손실은 손실 폭만큼만 감점
                    reward += base_score

            # 분석용 데이터 보존 (기존 변수 유지)
            regret = max(0.0, self.mfe - actual_pnl)
            capture = (actual_pnl / self.mfe) if self.mfe > 0.001 else (1.0 if actual_pnl > 0 else 0.0)
            info["trade_MFE"] = self.mfe
            info["trade_MAE"] = self.mae
            info["trade_Regret"] = regret
            info["trade_Capture"] = capture
            info["trade_Entry_Disp"] = self.entry_disparity            
            # 개별 환경 로그 저장
            trade_data = {
                "run": CONFIG.get("run_idx", 1),
                "stage": CONFIG.get("stage", 1),
                "train_step": CONFIG.get("cum_start_step", 0), # 환경 내에서는 정확한 실시간 step을 알 수 없으므로 시작점 기록
                "coin": self.coin_id, "pnl": actual_pnl,
                "mfe": self.mfe, "mae": self.mae, "regret": regret, "capture": capture,
                "entry_disp": self.entry_disparity, "is_counter": self.is_counter,
                "reason": info.get("trade_Exit_Reason", "Closed"), "score": float(reward)
            }
            pd.DataFrame([trade_data]).to_csv(self.env_log_path, mode='a', header=not os.path.exists(self.env_log_path), index=False)

        if self._ep_step_cnt >= self._max_ep_steps: done = True
        return obs, reward, done, truncated, info

def make_env(rank):
    return lambda: V33_FullHistoryEnv(coin_list=FULL_UNIVERSE, data_dir=DATA_DIR, rank=rank)

if __name__ == "__main__":
    train_env = VecFrameStack(SubprocVecEnv([make_env(i) for i in range(N_ENVS)]), n_stack=N_STACK)
    eval_env = VecFrameStack(DummyVecEnv([make_env(i + 100) for i in range(1)]), n_stack=N_STACK)
    
    policy_kwargs = dict(features_extractor_class=TCN6LayerExtractor, features_extractor_kwargs=dict(features_dim=256), net_arch=dict(pi=[128, 64], vf=[256, 128]))
    
    # [이어하기 로직] 제일 점수가 높은 best_score 모델을 동적으로 찾아서 불러옴
    best_score_files = glob.glob(os.path.join(MODEL_DIR, "v33_2_best_score_*.zip"))
    load_path = None
    
    if best_score_files:
        def extract_score(f):
            try:
                score_str = os.path.basename(f).replace("v33_2_best_score_", "").replace(".zip", "")
                return float(score_str)
            except:
                return -np.inf
                
        best_score_files.sort(key=extract_score, reverse=True)
        load_path = best_score_files[0]
        print(f"[*] Dynamically selected best score model: {load_path}")
    elif os.path.exists(RESUME_MODEL):
        load_path = RESUME_MODEL
    if load_path:
        print(f"[*] Fine-tuning from best model: {load_path} (LR: {LEARNING_RATE})")
        # [Fix] 크기 불일치(embedding mismatch) 해결을 위해 모델 생성 후 파라미터만 강제 로드
        model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, device="cuda", learning_rate=LEARNING_RATE)
        
        # 가중치 파일 열기
        _, params, _ = load_from_zip_file(load_path, device="cuda")
        
        if "policy" in params:
            # [Fix] 크기가 맞지 않는 파라미터(특히 임베딩)를 필터링하여 에러 방지
            state_dict = params["policy"]
            model_state_dict = model.policy.state_dict()
            
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    elif "coin_embedding.weight" in k:
                        # [Fix] 임베딩 크기가 다를 경우, 기존 코인들에 대한 기억은 복사해서 보존
                        new_weight = model_state_dict[k].clone()
                        min_coins = min(v.shape[0], new_weight.shape[0])
                        new_weight[:min_coins, :] = v[:min_coins, :]
                        filtered_state_dict[k] = new_weight
                        print(f"  [Partial Load] Copied existing {min_coins} embeddings for {k}")
                    else:
                        print(f"  [Skip] Mismatched shape for {k}: {v.shape} vs {model_state_dict[k].shape}")
            
            model.policy.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.set_parameters(params, exact_match=False)
        print("[*] Weights loaded successfully (Mismatched layers skipped)")
    else:
        print("[*] Starting fresh training from V30 weights")
        model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1, device="cuda", learning_rate=LEARNING_RATE)
        _, params, _ = load_from_zip_file(BASE_MODEL, device="cuda")
        if "policy" in params: model.policy.load_state_dict(params["policy"], strict=False)
        else: model.set_parameters(params, exact_match=False)
        
    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
    top5_cb = Top5ModelCallback(eval_cb=eval_cb, save_path=MODEL_DIR)
    stop_cb = PFTargetStopCallback(target_pf=TARGET_PF)
    
    # EvalCallback 뒤에 top5_cb를 배치하여 평가 직후 저장하게 함
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, top5_cb, stop_cb, TradeLoggerCallback(), ProgressBarCallback(TOTAL_TIMESTEPS), CurriculumCallback()], reset_num_timesteps=True)
    model.save(os.path.join(MODEL_DIR, FINAL_MODEL))
