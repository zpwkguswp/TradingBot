"""
V32 From-Scratch PPO 훈련 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 데이터  : data_storage/{COIN}_5m_full.parquet (2020~현재 5분봉)
■ 분할    : Train=2020~2024 / Val=2025 / Test=2026~현재
■ 모델    : TCN6LayerExtractor + PPO (백지 초기화, 전이 없음)
■ 콜백    : EvalCallback – 50,000스텝마다 val_env 평가, 최우수 저장
■ 저장    : elite_weights/v32_best_model_2h.zip (자동 덮어쓰기)
■ 최종    : test_env(2026)에서 실전 PnL 출력
"""

import gc
import os
import random
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")

from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE  # 50-coin 유니버스 & coin_id 인덱스 일관성

# ──────────────────────────────────────────────────────────────────
#  경로 / 상수
# ──────────────────────────────────────────────────────────────────
DATA_DIR  = "data_storage"
ELITE_DIR = "elite_weights"
LOG_DIR   = "v32_logs"
os.makedirs(ELITE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

DEVICE  = "cuda" if th.cuda.is_available() else "cpu"
N_STACK = 4          # VecFrameStack 프레임 수
TIMEFRAME = "5m"     # 수집 완료된 5분봉 전체 히스토리 사용
FULL_FILE_SUFFIX = "_5m_full.parquet"

# 저장 경로 (EvalCallback이 이 경로에 best_model.zip을 생성함)
BEST_MODEL_SAVE_DIR  = ELITE_DIR
BEST_MODEL_SAVE_NAME = "v32_best_model_2h"   # .zip 자동 추가
BEST_MODEL_PATH      = os.path.join(ELITE_DIR, f"{BEST_MODEL_SAVE_NAME}.zip")

# ──────────────────────────────────────────────────────────────────
#  시계열 분할 기준 (datetime, UTC 기준)
# ──────────────────────────────────────────────────────────────────
TRAIN_START = pd.Timestamp("2020-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
VAL_START   = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END     = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START  = pd.Timestamp("2026-01-01", tz="UTC")

# ──────────────────────────────────────────────────────────────────
#  V30 환경 하이퍼파라미터 (Trial-59 검증 완료 값 유지)
# ──────────────────────────────────────────────────────────────────
ENV_PARAMS = dict(
    target_profit = 0.008,
    far_th        = 2.2044178686597258,
    sl_atr_coef   = 3.8350847145336115,
    adx_th        = 0.169327336793444,
    max_disp      = 0.0587715969881679,
    trail_act     = 0.020,
)


# ══════════════════════════════════════════════════════════════════
#  1. 시계열 슬라이스 환경 래퍼
# ══════════════════════════════════════════════════════════════════
def _slice_df_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    timestamp 컬럼을 datetime으로 변환 후 split 구간만 잘라서 반환.
    timestamp가 UNIX ms이면 자동 변환.
    """
    if df.empty:
        return df

    ts = df["timestamp"].copy()

    # ms → datetime 변환 (값이 1e12 이상이면 ms 단위)
    if ts.dtype != "datetime64[ns, UTC]" and ts.dtype != "object":
        if ts.iloc[0] > 1e12:
            ts = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(ts, unit="s", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)

    if split == "train":
        mask = (ts >= TRAIN_START) & (ts <= TRAIN_END)
    elif split == "val":
        mask = (ts >= VAL_START) & (ts <= VAL_END)
    elif split == "test":
        mask = ts >= TEST_START
    else:
        mask = pd.Series([True] * len(df), index=df.index)

    sliced = df[mask].reset_index(drop=True)
    return sliced


class V32_TimeSliceEnv(V29_Universal_Env):
    """
    V29_Universal_Env를 상속하여 split 구간의 데이터만 사용하는 래퍼.
    에피소드마다 coin_list에서 랜덤 코인을 선택 (Cross-Sectional Shuffling).
    """

    def __init__(self, coin_list: list, data_dir: str, split: str, **kwargs):
        self.data_dir   = data_dir
        self.split_type_v32 = split   # 'train' / 'val' / 'test'

        # 부모 클래스를 안전한 BTC로 초기화 (첫 크래시 방지)
        super().__init__(
            data_dir   = data_dir,
            coin_files = [f"BTCUSDT{FULL_FILE_SUFFIX}"],
            coin_id    = 0,
            split_type = None,        # 부모의 split_type 비활성화 (우리가 직접 슬라이싱)
            **kwargs,
        )

        # ── 전체 코인 데이터를 슬라이스해서 캐싱 ──────────────────
        valid_coins = []
        for ticker in coin_list:
            key  = f"{ticker}{FULL_FILE_SUFFIX}"
            path = os.path.join(data_dir, key)
            if not os.path.exists(path):
                continue
            try:
                raw_df = pd.read_parquet(path).reset_index(drop=True)
            except Exception:
                continue

            sliced = _slice_df_by_split(raw_df, split)

            # V26.3 호환 특성 보정
            if "ema_20" not in sliced.columns:
                sliced["ema_20"] = sliced["close"].ewm(span=20, adjust=False).mean()
            if "ema_60" not in sliced.columns:
                sliced["ema_60"] = sliced["close"].ewm(span=60, adjust=False).mean()
            if "v26_disparity" not in sliced.columns:
                sliced["v26_disparity"] = (
                    (sliced["close"] - sliced["ema_60"]) / sliced["ema_60"]
                ) * 100.0

            if len(sliced) > 200:
                self.cached_dfs[key] = sliced
                valid_coins.append(ticker)
            else:
                print(f"  [Skip] {ticker} ({split}): 슬라이스 후 데이터 부족 ({len(sliced)}행)")

        self.coin_list_v32 = valid_coins if valid_coins else ["BTCUSDT"]
        print(f"  [V32 Env | {split}] 유효 코인: {len(self.coin_list_v32)}개")

        # ── 에피소드 최대 길이 제한 (속도 최적화 핵심) ────────────
        # 5분봉 5000스텝 ≈ 17일치 캔들. GPU 활용률을 높이고 학습 다양성 증가.
        self._max_ep_steps = 5000
        self._ep_step_cnt  = 0

    # ── 에피소드마다 랜덤 코인 선택 ───────────────────────────────
    def reset(self, seed=None, options=None):
        ticker  = random.choice(self.coin_list_v32)
        coin_id = FULL_UNIVERSE.index(ticker) if ticker in FULL_UNIVERSE else 0
        key     = f"{ticker}{FULL_FILE_SUFFIX}"

        if key in self.cached_dfs:
            self.coin_id      = coin_id
            self.coin_files   = [key]
            self.current_coin = key
            self.df           = self.cached_dfs[key]

        self._ep_step_cnt = 0   # 에피소드 스텝 카운터 초기화
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self._ep_step_cnt += 1
        if self._ep_step_cnt >= self._max_ep_steps:
            done = True   # 최대 스텝 도달 시 강제 종료
        return obs, reward, done, truncated, info


# ══════════════════════════════════════════════════════════════════
#  2. TCN 아키텍처 (v30_train.py와 동일)
# ══════════════════════════════════════════════════════════════════
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.padding = (ks - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, ks,
                                 padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCN6LayerExtractor(BaseFeaturesExtractor):
    """6-Layer Dilated TCN + Coin Embedding (백지 초기화 버전)."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.obs_dim   = 36
        self.coin_dim  = 1
        self.total_dim = self.obs_dim + self.coin_dim   # 37

        self.tcn = nn.Sequential(
            CausalConv1d(self.obs_dim, 16,  12, 1),  nn.ReLU(), nn.BatchNorm1d(16),
            CausalConv1d(16,  32,   8, 2),            nn.ReLU(), nn.BatchNorm1d(32),
            CausalConv1d(32,  64,   5, 4),            nn.ReLU(), nn.BatchNorm1d(64),
            CausalConv1d(64,  128,  3, 8),            nn.ReLU(), nn.BatchNorm1d(128),
            CausalConv1d(128, 256,  3, 16),           nn.ReLU(), nn.BatchNorm1d(256),
            CausalConv1d(256, 256,  2, 32),           nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        n_coins = max(len(FULL_UNIVERSE), 10)
        self.coin_embedding = nn.Embedding(num_embeddings=n_coins, embedding_dim=16)
        self.fc = nn.Linear(256 + 16, features_dim)

    def forward(self, observations):
        x       = observations.view(-1, N_STACK, self.total_dim)
        tcn_out = self.tcn(x[:, :, :self.obs_dim].transpose(1, 2))
        coin_id = x[:, 0, self.obs_dim].long()
        emb_out = self.coin_embedding(coin_id)
        return self.fc(th.cat([tcn_out, emb_out], dim=1))


# ══════════════════════════════════════════════════════════════════
#  3. PPO 하이퍼파라미터
# ══════════════════════════════════════════════════════════════════
POLICY_KWARGS = dict(
    features_extractor_class  = TCN6LayerExtractor,
    features_extractor_kwargs = dict(features_dim=256),
    net_arch = dict(pi=[128, 64], vf=[256, 128]),
)

PPO_KWARGS = dict(
    verbose       = 0,
    policy_kwargs = POLICY_KWARGS,
    learning_rate = 2e-5,
    ent_coef      = 0.001,
    max_grad_norm = 0.5,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 5,
    gamma         = 0.99,
    clip_range    = 0.2,
    device        = DEVICE,
)


# ══════════════════════════════════════════════════════════════════
#  4. 유틸: 파일 필터 & 환경 팩토리
# ══════════════════════════════════════════════════════════════════
def _filter_coins(coin_list: list, split: str, min_rows: int = 200) -> list:
    """슬라이스 후 데이터가 충분한 코인만 반환."""
    valid = []
    for ticker in coin_list:
        path = os.path.join(DATA_DIR, f"{ticker}{FULL_FILE_SUFFIX}")
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_parquet(path)
            sliced = _slice_df_by_split(raw, split)
            if len(sliced) >= min_rows:
                valid.append(ticker)
        except Exception:
            pass
    return valid


def make_env_fn(coin_list: list, split: str):
    """DummyVecEnv용 env 생성 함수 반환."""
    def _init():
        return V32_TimeSliceEnv(
            coin_list = coin_list,
            data_dir  = DATA_DIR,
            split     = split,
            **ENV_PARAMS,
        )
    return _init


def build_venv(coin_list: list, split: str, n_envs: int = 1):
    """DummyVecEnv + VecFrameStack으로 묶인 환경 반환."""
    fns  = [make_env_fn(coin_list, split) for _ in range(n_envs)]
    venv = DummyVecEnv(fns)
    return VecFrameStack(venv, n_stack=N_STACK)


# ══════════════════════════════════════════════════════════════════
#  5. 실전 테스트 (Test Set 평가)
# ══════════════════════════════════════════════════════════════════
def run_test_evaluation(model_path: str, coin_list: list):
    """
    저장된 best_model을 불러와 test_env(2026년)에서 한 에피소드 실행 후
    최종 PnL(수익률)을 터미널에 출력합니다.
    """
    print("\n" + "=" * 60)
    print("  [V32] 실전 테스트 시작 (Test Set: 2026년~현재)")
    print("=" * 60)

    test_coins = _filter_coins(coin_list, "test", min_rows=10)
    if not test_coins:
        print("  [Warning] Test Set 데이터 없음 (2026년 이후 데이터 부족). 스킵.")
        return

    print(f"  테스트 가능 코인: {len(test_coins)}개")

    # 모델 로드
    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    model = PPO.load(model_path, device=DEVICE, custom_objects=custom_objects)
    print(f"  모델 로드 완료: {model_path}")

    test_venv = build_venv(test_coins, "test", n_envs=1)

    obs = test_venv.reset()
    done_flags  = np.zeros(test_venv.num_envs, dtype=bool)
    bot_returns = np.zeros(test_venv.num_envs)
    peak_eq     = np.ones(test_venv.num_envs)
    mdds        = np.zeros(test_venv.num_envs)
    gross_p = gross_l = 0.0
    total_trades = 0

    while not done_flags.all():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = test_venv.step(action)

        for i, info in enumerate(infos):
            if done_flags[i]:
                continue
            if dones[i]:
                done_flags[i] = True

            if info.get("trade_closed"):
                pnl = info.get("trade_Actual_PnL", 0.0)
                total_trades += 1
                if pnl > 0:
                    gross_p += pnl
                else:
                    gross_l += abs(pnl)

            eq = info.get("bot_equity", 1.0)
            if eq > peak_eq[i]:
                peak_eq[i] = eq
            dd = (peak_eq[i] - eq) / (peak_eq[i] + 1e-9)
            if dd > mdds[i]:
                mdds[i] = dd
            bot_returns[i] = eq - 1.0

    pf = gross_p / (gross_l + 1e-9)

    print("\n  ─── 실전 테스트 결과 (Test Set: 2026~현재) ───")
    print(f"  테스트 코인 수  : {len(test_coins)}")
    print(f"  총 거래 수      : {total_trades}")
    print(f"  평균 수익률     : {bot_returns.mean() * 100:.2f}%")
    print(f"  평균 MDD        : {mdds.mean() * 100:.2f}%")
    print(f"  Profit Factor   : {pf:.4f}")
    print("  ─────────────────────────────────────────────")

    test_venv.close()
    del model
    gc.collect()
    th.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════
#  6. 메인 훈련 실행
# ══════════════════════════════════════════════════════════════════
def run_v32_from_scratch():
    print("\n" + "=" * 60)
    print("  V32 From-Scratch PPO 훈련 개시")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  N_STACK    : {N_STACK}")
    print(f"  Train      : 2020-01-01 ~ 2024-12-31")
    print(f"  Val        : 2025-01-01 ~ 2025-12-31")
    print(f"  Test       : 2026-01-01 ~ 현재")
    print(f"  Best Model : {BEST_MODEL_PATH}")
    print("=" * 60 + "\n")

    # ── 코인 필터링 ───────────────────────────────────────────────
    print("[Step 1] 코인 데이터 필터링 중...")
    train_coins = _filter_coins(FULL_UNIVERSE, "train", min_rows=500)
    val_coins   = _filter_coins(FULL_UNIVERSE, "val",   min_rows=200)
    test_coins  = _filter_coins(FULL_UNIVERSE, "test",  min_rows=10)

    print(f"  Train 코인: {len(train_coins)}개")
    print(f"  Val   코인: {len(val_coins)}개")
    print(f"  Test  코인: {len(test_coins)}개")

    if not train_coins:
        print("[Fatal] 훈련 가능한 코인 없음. 종료.")
        return
    if not val_coins:
        print("[Warning] Validation 코인 없음. EvalCallback 없이 훈련합니다.")

    # ── 환경 생성 ─────────────────────────────────────────────────
    print("\n[Step 2] 환경 생성 중...")

    # 훈련 환경: DummyVecEnv는 순차 실행이므로 4개가 최적
    # (8~16으로 늘려도 DummyVecEnv에서는 벽시계 시간 동일)
    N_TRAIN_ENVS = min(4, len(train_coins))
    N_VAL_ENVS   = 1   # EvalCallback은 단일 환경 권장

    print(f"  훈련 환경 수: {N_TRAIN_ENVS}")
    train_venv = build_venv(train_coins, "train", n_envs=N_TRAIN_ENVS)

    eval_callback = None
    if val_coins:
        val_venv = build_venv(val_coins, "val", n_envs=N_VAL_ENVS)

        # ── EvalCallback 설정 ─────────────────────────────────────
        # eval_freq: 훈련 venv 기준 총 스텝 → 실제 환경 1개 기준으로는 /N_TRAIN_ENVS
        # SB3 EvalCallback의 eval_freq는 num_timesteps 기준이므로 그대로 사용
        eval_callback = EvalCallback(
            eval_env          = val_venv,
            best_model_save_path = BEST_MODEL_SAVE_DIR,
            log_path          = LOG_DIR,
            eval_freq         = max(200_000 // N_TRAIN_ENVS, 1000),  # 200k 글로벌 스텝마다
            n_eval_episodes   = 15,   # 48개 전체 → 15개 샘플 평가
            deterministic     = True,
            render            = False,
            verbose           = 1,
        )
        # EvalCallback은 자동으로 {best_model_save_path}/best_model.zip 저장
        # 이후에 v32_best_model_2h.zip으로 복사
        print(f"  EvalCallback: 50,000스텝마다 {len(val_coins)}개 코인 평가")
    else:
        val_venv = None
        print("  EvalCallback 비활성화 (Val 데이터 없음)")

    # ── 모델 완전 신규 초기화 (From Scratch) ─────────────────────
    print("\n[Step 3] PPO 모델 초기화 (From Scratch)...")
    model = PPO("MlpPolicy", train_venv, **PPO_KWARGS)

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"  파라미터 수: {param_count:,}")
    print(f"  아키텍처  : TCN6Layer + CoinEmbedding(dim=16) + MLP[128,64]")
    print("  사전 가중치 없음 → 완전 백지 훈련")

    # ── 훈련 ──────────────────────────────────────────────────────
    print("\n[Step 4] 훈련 시작 (total_timesteps=3,000,000)...")
    print("  (EvalCallback이 검증 수익 최고치 경신 시 자동 저장)\n")

    try:
        model.learn(
            total_timesteps    = 3_000_000,
            callback           = eval_callback,
            reset_num_timesteps= True,
            progress_bar       = False,
        )
    except KeyboardInterrupt:
        print("\n  [!] 사용자 중단. 현재 상태 저장 후 종료합니다.")

    # ── 최종 모델 저장 ────────────────────────────────────────────
    final_path = os.path.join(ELITE_DIR, "v32_final")
    model.save(final_path)
    print(f"\n[Step 5] 최종 모델 저장: {final_path}.zip")

    # EvalCallback이 저장한 best_model.zip → v32_best_model_2h.zip으로 복사
    eval_best_src = os.path.join(BEST_MODEL_SAVE_DIR, "best_model.zip")
    if os.path.exists(eval_best_src):
        import shutil
        shutil.copy(eval_best_src, BEST_MODEL_PATH)
        print(f"  Best Model 복사 완료: {eval_best_src} → {BEST_MODEL_PATH}")
    else:
        # EvalCallback 없이 훈련된 경우 최종 모델을 best로 사용
        model.save(os.path.join(ELITE_DIR, BEST_MODEL_SAVE_NAME))
        print(f"  (EvalCallback 저장 없음) 최종 모델을 best로 지정: {BEST_MODEL_PATH}")

    # ── 환경 정리 ─────────────────────────────────────────────────
    train_venv.close()
    if val_venv is not None:
        val_venv.close()
    del model
    gc.collect()
    th.cuda.empty_cache()

    # ── 실전 테스트 ───────────────────────────────────────────────
    print("\n[Step 6] 실전 테스트 (Test Set: 2026년~현재)")
    if os.path.exists(BEST_MODEL_PATH):
        run_test_evaluation(BEST_MODEL_PATH, test_coins if test_coins else FULL_UNIVERSE)
    else:
        print(f"  [Warning] Best model 파일 없음: {BEST_MODEL_PATH}")

    print("\n  V32 훈련 파이프라인 완료!")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_v32_from_scratch()
