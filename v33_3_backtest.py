import os
import warnings
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from datetime import datetime

warnings.filterwarnings("ignore")

# 기존 프로젝트 구조에서 임포트
from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE, TCN6LayerExtractor

# ── [설정] 훈련장과 100% 동일하게 세팅 ───────────────────────
MODEL_PATH  = "elite_weights/v33_3_snapshots/stage3_run018_score31.3_20260503_180454/model.zip"
DATA_DIR    = "data_storage"
N_STACK     = 4
N_COINS     = 50
N_EPISODES  = 10
MAX_STEPS   = 10000

# [🌟 중요] 훈련 시 사용했던 환경 파라미터와 100% 동치
TEST_ENV_PARAMS = dict(
    target_profit = 0.008,
    far_th        = 10.0,
    sl_atr_coef   = 3.835,
    adx_th        = 0.0,
    max_disp      = 10.0,
    trail_act     = 0.020,
)

FULL_FILE_SUFFIX = "_5m_full.parquet"

# ── 테스트 환경 (훈련장 거울 동기화) ──────────────────────────
class V33_TestEnv(V29_Universal_Env):
    def __init__(self, coin_list, data_dir, **kwargs):
        self.data_dir      = data_dir
        self.full_universe = coin_list
        init_coin = coin_list[0]
        
        # 훈련 환경(V29/V28)의 물리법칙을 그대로 상속
        super().__init__(
            data_dir   = data_dir,
            coin_files = [f"{init_coin}{FULL_FILE_SUFFIX}"],
            coin_id    = 0,
            split_type = None, # 전체 데이터 로드 후 수동 슬라이싱
            **kwargs
        )
        self._ep_step_cnt  = 0
        self._max_ep_steps = MAX_STEPS
        self._trades       = []

    def _load_coin_data(self, coin):
        path = os.path.join(self.data_dir, f"{coin}{FULL_FILE_SUFFIX}")
        if not os.path.exists(path): return None
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # [🌟 동기화 1] 훈련장 전용 지표 해킹 로직 (Mirroring)
        df["returns"]    = df["close"].pct_change().fillna(0)
        df["log_volume"] = np.log1p(df["volume"])
        df["hl_ratio"]   = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        for s in [5, 10, 20, 60]:
            df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
        for s in [20, 60, 200]:
            df[f"h1_ema_{s}"] = df["close"].ewm(span=s * 12, adjust=False).mean()
            
        # 훈련 시 적용했던 필터 무력화용 가짜 지표들
        df["actual_h1_ema_200"] = df["h1_ema_200"].copy()
        df["h1_ema_200"] = df["close"] 
            
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)
        df["atr_raw"] = df["atr_raw"].fillna(0.01).clip(0.001, 0.1)
        df["structural_rev_long"] = 1.0
        df["structural_rev_short"] = 1.0
        df["ema_squeeze"] = 0.0
        df["adx_14"] = 30.0
        
        # V28 상위 호환용 매크로 지표
        df["macro_adx"] = 100.0
        df["macro_atr_ratio"] = 1.0
        df["macro_bb_width"] = 1.0
        
        # [🌟 구간 설정] 마지막 10% (훈련에 쓰이지 않은 미래 데이터)
        n = len(df)
        cut = int(n * 0.90)
        return df.iloc[cut:].reset_index(drop=True)

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        # 37차원 보정 (훈련 환경 규격 준수)
        if len(obs) == 33:
            padding = np.zeros(4, dtype=np.float32)
            return np.concatenate([obs, padding])
        return obs

    def reset(self, seed=None, options=None):
        self._ep_step_cnt = 0
        coin = np.random.choice(self.full_universe)
        df   = self._load_coin_data(coin)
        if df is None:
            coin = self.full_universe[0]
            df   = self._load_coin_data(coin)
        self.df = df
        self.coin_id = self.full_universe.index(coin) if coin in self.full_universe else 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # [🌟 동기화 2] 모든 자체 로직/우회 필터를 삭제하고 부모에게 100% 위임
        # 이 시점에서 진입 문턱(hold_signal_thr) 등도 부모의 설정(0.05 등)을 따릅니다.
        was_in = getattr(self, 'position', None) is not None
        obs, reward, done, truncated, info = super().step(action)
        self._ep_step_cnt += 1
        
        if self._ep_step_cnt >= self._max_ep_steps: 
            done = True
            
        now_in = getattr(self, 'position', None) is not None
        # 거래 내역 수집 (부모 환경에서 리턴된 데이터 활용)
        if was_in and not now_in:
            init = 10000.0
            eq = float(getattr(self, 'balance', init)) / init
            self._trades.append({
                "pnl": float(info.get("trade_Actual_PnL", 0.0)),
                "reason": str(info.get("trade_Exit_Reason", "Unknown")),
                "equity": round(eq, 6)
            })
        return obs, reward, done, truncated, info

def compute_metrics(trades):
    if not trades: return None
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["pnl"] > 0]
    wr = len(wins) / total * 100
    gp = wins["pnl"].sum() * 100; gl = abs(df[df["pnl"] < 0]["pnl"].sum()) * 100
    pf = gp / gl if gl > 0 else float("inf")
    return dict(total=total, wr=wr, pf=pf, net=gp-gl)

def main():
    print("=" * 64)
    print("  [V33-3 백테스트] Run 18 모델 → 완전 거울 동기화 테스트")
    print("  (모든 우회 필터 및 자체 로직 제거)")
    print("=" * 64)
    
    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
    print(f"  ✅ 모델 로드 성공: {MODEL_PATH}")
    
    all_trades = []
    for ep_idx in range(N_EPISODES):
        def _make(): return V33_TestEnv(coin_list=FULL_UNIVERSE, data_dir=DATA_DIR, **TEST_ENV_PARAMS)
        venv = VecFrameStack(DummyVecEnv([_make]), n_stack=N_STACK)
        obs = venv.reset()
        done_flags = [False]
        env_ref = venv.envs[0]
        
        while not all(done_flags):
            # [🌟 중요] 결정론적 모드로 모델의 핵심 판단만 평가
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_flags, _ = venv.step(action)
            
        all_trades.extend(env_ref._trades)
        venv.close()
        print(f"  Episode {ep_idx+1}/{N_EPISODES}: {len(env_ref._trades)}건 거래")
        
    m = compute_metrics(all_trades)
    if not m:
        print("\n  ⚠️ 거래 0건: 훈련장과 똑같은 조건에서는 미래 데이터 진입에 실패했습니다."); return
        
    print(f"\n{'='*64}\n  ── 백테스트 결과 (Mirror Test) ─────────────\n{'='*64}")
    print(f"  {'매매횟수':<14} {m['total']:>8,} 회\n  {'승률':<14} {m['wr']:>8.2f} %\n  {'PF':<14} {m['pf']:>8.4f}\n  {'누적 수익률':<13} {m['net']:>+8.2f} %")
    print(f"{'='*64}")

if __name__ == "__main__":
    main()
