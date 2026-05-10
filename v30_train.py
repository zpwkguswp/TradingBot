"""
V30 대함대 훈련 엔진 (Universal Alpha Fleet)

원칙 1: Volatility-Normalized Reward  - ATR로 나눈 보상 지급
원칙 2: PPO 방어막                   - ent_coef, grad_clip, embedding weight_decay
원칙 3: Curriculum Learning           - BTC/ETH  Top15  50코인 3단계
원칙 4: Cross-Sectional Batching      - 에피소드마다 랜덤 코인 셔플

"""
import gc
import os
import random
import time
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore")
from v29_env import V29_Universal_Env

#  경로 
DATA_DIR  = "data_storage"
ELITE_DIR = "elite_weights"
LOG_DIR   = "v30_logs"
os.makedirs(ELITE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE  = "cuda" if th.cuda.is_available() else "cpu"
N_STACK = 4

#  커리큘럼 3단계 코인 리스트 
STAGE_1_COINS = ["BTCUSDT", "ETHUSDT"]
STAGE_2_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
                 "BNBUSDT", "DOGEUSDT", "DOTUSDT", "POLUSDT",
                 "LTCUSDT", "NEARUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"]
STAGE_3_COINS = STAGE_2_COINS + [
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SEIUSDT",
    "STXUSDT", "TIAUSDT", "SUIUSDT", "ORDIUSDT", "WIFUSDT",
    "PENDLEUSDT", "JUPUSDT", "PYTHUSDT", "RENDERUSDT", "FETUSDT",
    "WLDUSDT", "BLURUSDT", "GMXUSDT", "DYDXUSDT", "RUNEUSDT",
    "ATOMUSDT", "ALGOUSDT", "EGLDUSDT", "HBARUSDT", "FLOWUSDT",
    "AXSUSDT", "SANDUSDT", "MANAUSDT", "CHZUSDT",
    "APEUSDT", "GALAUSDT", "IMXUSDT", "LRCUSDT", "ZILUSDT",
][:50]  # 최대 50개

# 전체 유니버스 인덱스 맵 (coin_id 일관성)
FULL_UNIVERSE = STAGE_3_COINS

TIMEFRAME = "2h"

#  V30 환경 하이퍼파라미터 
ENV_PARAMS = dict(
    target_profit=0.008,
    far_th=2.2044178686597258,
    sl_atr_coef=3.8350847145336115,
    adx_th=0.169327336793444,
    max_disp=0.0587715969881679,
    trail_act=0.020,
)


# 
# 원칙 4: Cross-Sectional (Shuffling) 환경 래퍼
# 
class ShuffledCoinEnv(V29_Universal_Env):
    """
    에피소드가 끝날 때마다 coin_list에서 랜덤으로 하나를 선택합니다.
    핵심 수정: __init__ 시 모든 코인 df를 cached_dfs에 미리 로드해두고,
    reset() 때 current_coin만 교체하여 KeyError를 방지합니다.
    """
    def __init__(self, coin_list: list, data_dir: str, **kwargs):
        self.data_dir  = data_dir
        
        # 무조건 안전한 BTCUSDT로 부모 클래스를 초기화하여 초기 크래시 방지
        super().__init__(
            data_dir=data_dir,
            coin_files=[f"BTCUSDT_{TIMEFRAME}.parquet"],
            coin_id=0,
            **kwargs,
        )

        valid_coins = []
        split = kwargs.get("split_type", None)
        
        for t in coin_list:
            key = f"{t}_{TIMEFRAME}.parquet"
            p = os.path.join(data_dir, key)
            if not os.path.exists(p):
                continue
            
            try:
                df = pd.read_parquet(p).reset_index(drop=True)
            except Exception:
                continue

            # [V26.3 호환성] 부모 클래스에서 필요한 특성 수동 계산
            if 'ema_20' not in df.columns:
                df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            if 'ema_60' not in df.columns:
                df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
            if 'v26_disparity' not in df.columns:
                df['v26_disparity'] = ((df['close'] - df['ema_60']) / df['ema_60']) * 100.0

            if split in ("train", "eval"):
                cut = int(len(df) * 0.8)
                df = df.iloc[:cut] if split == "train" else df.iloc[cut:]
                df = df.reset_index(drop=True)
                
            if len(df) > 10:
                self.cached_dfs[key] = df
                valid_coins.append(t)
            else:
                print(f"    [Warning] {t} 데이터 부족(길이 {len(df)})으로 환경 편입 스킵.")
                
        # 필터링된 코인 리스트만 사용하도록 교체 (KeyError 및 IndexError 방지)
        self.coin_list = valid_coins if valid_coins else ["BTCUSDT"]

    def _random_pick(self):
        ticker  = random.choice(self.coin_list)
        coin_id = FULL_UNIVERSE.index(ticker) if ticker in FULL_UNIVERSE else 0
        return ticker, coin_id

    def reset(self, seed=None, options=None):
        ticker, coin_id = self._random_pick()
        key = f"{ticker}_{TIMEFRAME}.parquet"

        # cached_dfs에 있는 경우에만 코인 교체
        if key in self.cached_dfs:
            self.coin_id     = coin_id
            self.coin_files  = [key]
            self.current_coin = key          # 부모 reset()이 이 키로 df를 찾음
            self.df           = self.cached_dfs[key]

        return super().reset(seed=seed, options=options)


# 
# 원칙 2: TCN + Embedding 아키텍처 (Weight Decay 지원)
# 
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.padding = (ks - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, ks, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCN6LayerExtractor(BaseFeaturesExtractor):
    """
    원칙 2: Embedding 레이어는 optimizer에서 weight_decay를 별도 적용합니다.
    (train loop에서 param_groups를 분리하여 L2 정규화 강제)
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.obs_dim   = 36
        self.coin_dim  = 1
        self.total_dim = self.obs_dim + self.coin_dim

        self.tcn = nn.Sequential(
            CausalConv1d(self.obs_dim, 16,  12, 1),  nn.ReLU(), nn.BatchNorm1d(16),
            CausalConv1d(16,  32,  8, 2),             nn.ReLU(), nn.BatchNorm1d(32),
            CausalConv1d(32,  64,  5, 4),             nn.ReLU(), nn.BatchNorm1d(64),
            CausalConv1d(64,  128, 3, 8),             nn.ReLU(), nn.BatchNorm1d(128),
            CausalConv1d(128, 256, 3, 16),            nn.ReLU(), nn.BatchNorm1d(256),
            CausalConv1d(256, 256, 2, 32),            nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        n_coins = 100 # [Fix] 고정 크기로 설정하여 코인 리스트 변경 시에도 에러 방지
        self.coin_embedding = nn.Embedding(num_embeddings=n_coins, embedding_dim=16)
        self.fc = nn.Linear(256 + 16, features_dim)

    def forward(self, observations):
        x = observations.view(-1, N_STACK, self.total_dim)
        tcn_out = self.tcn(x[:, :, :self.obs_dim].transpose(1, 2))
        coin_id = x[:, 0, self.obs_dim].long()
        emb_out = self.coin_embedding(coin_id)
        return self.fc(th.cat([tcn_out, emb_out], dim=1))


def build_optimizer_with_embedding_decay(model, lr, weight_decay_emb=1e-3):
    """
    원칙 2: Embedding 파라미터에만 L2 정규화(weight_decay)를 적용.
    나머지 파라미터는 weight_decay=0.
    """
    emb_params   = []
    other_params = []
    for name, param in model.policy.named_parameters():
        if "coin_embedding" in name:
            emb_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "weight_decay": 0.0},
        {"params": emb_params,   "weight_decay": weight_decay_emb},
    ]
    return th.optim.Adam(param_groups, lr=lr)


# 
# 원칙 1+2: ATR 정규화 보상 확인용 콜백
# 
class V30TrainCallback(BaseCallback):
    """훈련 진행 모니터링 및 최우수 모델 자동 저장."""
    SAVE_INTERVAL   = 100_000
    REPORT_INTERVAL =  10_000
    BANKRUPT_RATIO  = 0.55       # 원금 55% 이하  조기 종료

    def __init__(self, stage: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.stage          = stage
        self.save_path      = save_path
        self._last_save     = 0
        self._last_report   = 0
        self._best_score    = -np.inf
        self._trade_buf     = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # 조기 파산 감지
        balances = [info.get("balance", 10_000) for info in infos]
        if np.mean(balances) < 10_000 * self.BANKRUPT_RATIO:
            print(f"[V30-S{self.stage}]  파산 감지 @ step {self.num_timesteps}. 중단.")
            return False

        # 거래 로그 버퍼
        for info in infos:
            if info.get("trade_closed"):
                self._trade_buf.append({
                    "stage":   self.stage,
                    "step":    self.num_timesteps,
                    "pnl":     info.get("trade_Actual_PnL", 0.0),
                    "reason":  info.get("trade_Exit_Reason", ""),
                })

        # 주기적 저장 및 리포트
        if self.num_timesteps - self._last_save >= self.SAVE_INTERVAL:
            self._last_save = self.num_timesteps
            self._flush_logs()
            ckpt = f"{self.save_path}_stage{self.stage}_ckpt{self.num_timesteps}.zip"
            self.model.save(ckpt)
            print(f"[V30-S{self.stage}]  체크포인트 저장: {ckpt}")

        if self.num_timesteps - self._last_report >= self.REPORT_INTERVAL:
            self._last_report = self.num_timesteps
            equities = [info.get("bot_equity", 1.0) for info in infos]
            print(f"[V30-S{self.stage}] step={self.num_timesteps:,} | "
                  f"avg_equity={np.mean(equities):.3f} | "
                  f"trades_buf={len(self._trade_buf)}")

        return True

    def _flush_logs(self):
        if not self._trade_buf:
            return
        path = os.path.join(LOG_DIR, f"v30_stage{self.stage}_trades.csv")
        df   = pd.DataFrame(self._trade_buf)
        hdr  = not os.path.exists(path)
        df.to_csv(path, mode="a", index=False, header=hdr)
        self._trade_buf = []

    def on_training_end(self):
        self._flush_logs()


# 
# 환경 팩토리
# 
def _filter_existing(coin_list):
    """파케이 파일이 존재하고, 데이터가 비어있지 않은(최소 10줄 이상) 코인만 반환."""
    out = []
    for t in coin_list:
        p = os.path.join(DATA_DIR, f"{t}_{TIMEFRAME}.parquet")
        if os.path.exists(p):
            try:
                df_temp = pd.read_parquet(p)
                if len(df_temp) > 100:  # Train/Eval 스플릿을 고려하여 최소 100줄 이상 필수
                    out.append(t)
                else:
                    print(f"    데이터 부족 (길이 {len(df_temp)}), 스킵: {t}")
            except Exception as e:
                print(f"    파일 읽기 에러, 스킵: {t} ({e})")
        else:
            print(f"    파케이 없음, 스킵: {t}")
    return out


def make_shuffled_env(coin_list, split):
    def _init():
        return ShuffledCoinEnv(
            coin_list=coin_list,
            data_dir=DATA_DIR,
            split_type=split,
            **ENV_PARAMS,
        )
    return _init


def make_venv(coin_list, split, n_envs):
    fns = [make_shuffled_env(coin_list, split) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns)
    return VecFrameStack(venv, n_stack=N_STACK)


# 
# PPO 기본 설정 (원칙 2)
# 
POLICY_KWARGS = dict(
    features_extractor_class=TCN6LayerExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[128, 64], vf=[256, 128]),
)

PPO_BASE = dict(
    verbose       = 0,
    policy_kwargs = POLICY_KWARGS,
    learning_rate = 2e-5,
    # 🚨 0.015였던 족쇄를 0.0001로 대폭 완화! (차트를 보고 확신을 갖도록 허락)
    ent_coef      = 0.0001,      
    max_grad_norm = 0.5,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 5,
    gamma         = 0.99,
    clip_range    = 0.2,
    device        = DEVICE,
)


# 
# 커리큘럼 3단계 훈련 (원칙 3)
# 
def train_stage(stage: int, coin_list, n_envs, total_steps,
                pretrained_path=None):
    """
    stage     : 1, 2, 3
    coin_list : 해당 스테이지에서 사용할 코인 목록
    pretrained_path : 이전 스테이지 모델 경로 (전이 학습)
    """
    coins = _filter_existing(coin_list)
    if not coins:
        print(f"[V30-S{stage}]  사용 가능한 파케이 없음. 스킵.")
        return None

    print(f"\n{'='*60}")
    print(f"[V30]  Stage {stage} 훈련 시작")
    print(f"  코인 수    : {len(coins)}")
    print(f"  병렬 환경  : {n_envs}")
    print(f"  총 스텝    : {total_steps:,}")
    print(f"  전이 모델  : {pretrained_path or '없음 (신규 훈련)'}")
    print(f"{'='*60}")

    save_base = os.path.join(ELITE_DIR, f"v30_stage{stage}")
    final_path = f"{save_base}_final"

    # 이미 완료된 스테이지라면 스킵하고 경로만 반환 (Resume 지원)
    if os.path.exists(f"{final_path}.zip"):
        print(f"[V30-S{stage}]  이미 훈련이 완료된 모델이 존재합니다. 훈련을 건너뛰고 다음 단계로 진행합니다.")
        return final_path

    venv = make_venv(coins, "train", n_envs)

    # 1. 모델과 커스텀 Optimizer 먼저 생성
    model = PPO("MlpPolicy", venv, **PPO_BASE)
    try:
        model.policy.optimizer = build_optimizer_with_embedding_decay(
            model, lr=model.learning_rate, weight_decay_emb=1e-3
        )
        print(f"[V30-S{stage}]  Embedding weight_decay=1e-3 적용")
    except Exception as e:
        print(f"[V30-S{stage}]  Embedding weight_decay 설정 실패 (무시): {e}")

    # 2. 파라미터 로드 (저장된 Optimizer StateDict의 param_groups와 완벽 일치하게 됨)
    if pretrained_path and os.path.exists(pretrained_path + ".zip"):
        safe_load_model_parameters(model, pretrained_path)
        # 전이 학습: learning_rate와 ent_coef를 살짝 낮춤
        model.learning_rate = 1e-5
        model.ent_coef      = 0.0001 # ✅ PPO_BASE와 동일하게 0.0001로 통일!
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = 1e-5
        print(f"[V30-S{stage}]  전이 학습 모드 (lr=1e-5, ent=0.0001)")

    callback = V30TrainCallback(stage=stage, save_path=save_base)
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=True)
    callback.on_training_end()

    final_path = f"{save_base}_final"
    model.save(final_path)
    print(f"[V30-S{stage}]  최종 모델 저장: {final_path}.zip")

    venv.close()
    del model
    gc.collect()
    th.cuda.empty_cache()

    return final_path


# 
# 파라미터 로드 헬퍼 (옵티마이저 미스매치 방어)
# 
def safe_load_model_parameters(model, path):
    """
    SB3 set_parameters는 옵티마이저 파라미터 그룹 수가 다르면 ValueError 발생.
    이 함수는 실패 시 가중치(policy.pth)만 강제 로드하여 훈련을 계속할 수 있게 함.
    """
    if not path:
        return
    full_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(full_path):
        return
        
    try:
        # 가급적 전체 상태(옵티마이저 포함) 로드 시도
        model.set_parameters(path)
    except ValueError as e:
        if "parameter groups" in str(e):
            print(f"    [RobustLoad] 옵티마이저 그룹 불일치 감지. 가중치(Weights)만 강제 로드합니다.")
            import zipfile
            import io
            with zipfile.ZipFile(full_path, "r") as archive:
                with archive.open("policy.pth") as f:
                    policy_weights = th.load(io.BytesIO(f.read()), map_location=DEVICE)
            model.policy.load_state_dict(policy_weights)
        else:
            raise e

# 
# 평가 루프
# 
def evaluate_model(model_path: str, coin_list, label="eval"):
    coins = _filter_existing(coin_list)
    if not coins:
        print("[V30 Eval] 코인 없음.")
        return {}

    venv = make_venv(coins, "eval", n_envs=min(len(coins), 4))
    # 평가 루프에서도 동일하게 커스텀 옵티마이저를 씌운 뒤 파라미터를 로드해야 에러가 안 남
    model = PPO("MlpPolicy", venv, **PPO_BASE)
    
    try:
        model.policy.optimizer = build_optimizer_with_embedding_decay(
            model, lr=model.learning_rate, weight_decay_emb=1e-3
        )
    except Exception:
        pass
    
    safe_load_model_parameters(model, model_path)

    obs = venv.reset()
    episodes_done = np.zeros(venv.num_envs)
    bot_rets      = np.zeros(venv.num_envs)
    peak_eq       = np.ones(venv.num_envs)
    mdds          = np.zeros(venv.num_envs)
    gross_p = gross_l = 0.0
    total_trades = 0

    while not (episodes_done >= 1).all():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        for i, info in enumerate(infos):
            if episodes_done[i] >= 1:
                continue
            if dones[i]:
                episodes_done[i] += 1
            if info.get("trade_closed"):
                pnl = info.get("trade_Actual_PnL", 0.0)
                total_trades += 1
                if pnl > 0: gross_p += pnl
                else:       gross_l += abs(pnl)
            eq = info.get("bot_equity", 1.0)
            if eq > peak_eq[i]: peak_eq[i] = eq
            dd = (peak_eq[i] - eq) / (peak_eq[i] + 1e-9)
            if dd > mdds[i]: mdds[i] = dd
            bot_rets[i] = eq - 1.0

    pf     = gross_p / (gross_l + 1e-9)
    result = {
        "label":      label,
        "coins":      len(coins),
        "trades":     total_trades,
        "profit_factor": round(float(pf), 4),
        "avg_return": round(float(bot_rets.mean()), 4),
        "avg_mdd":    round(float(mdds.mean()), 4),
    }
    print(f"\n[V30 Eval | {label}]")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    venv.close()
    del model
    gc.collect()
    th.cuda.empty_cache()
    return result


# 
# 메인 실행
# 
def run_v30_curriculum():
    """
    원칙 3: 3단계 커리큘럼 학습 실행
    각 단계가 완료되면 자동으로 다음 단계로 전이 학습.
    """
    print("\n" + "="*60)
    print("  V30 대함대 훈련 개시 (4-Principle Universal Alpha)")
    print("="*60)
    print(f"  Device : {DEVICE}")
    print(f"  원칙 1 : ATR 정규화 보상 (v29_env.py 내장)")
    print(f"  원칙 2 : ent_coef=0.0001, max_grad_norm=0.5, Emb WD=1e-3")
    print(f"  원칙 3 : 3단계 커리큘럼 (BTC/ETH -> Top15 -> 50 coins)")
    print(f"  원칙 4 : Cross-Sectional Shuffling (에피소드마다 코인 랜덤)")
    print("="*60 + "\n")

    #  Stage 1: BTC + ETH 기초 훈련 
    s1_path = train_stage(
        stage=1,
        coin_list=STAGE_1_COINS,
        n_envs=2,
        total_steps=500_000,
        pretrained_path=None,
    )

    if s1_path:
        evaluate_model(s1_path, STAGE_1_COINS, label="Stage1_eval")

    #  Stage 2: Top15 전이 학습 
    s2_path = train_stage(
        stage=2,
        coin_list=STAGE_2_COINS,
        n_envs=4,
        total_steps=1_000_000,
        pretrained_path=s1_path,
    )

    if s2_path:
        evaluate_model(s2_path, STAGE_2_COINS, label="Stage2_eval")

    #  Stage 3: 전체 50코인 파인튜닝 
    s3_path = train_stage(
        stage=3,
        coin_list=STAGE_3_COINS,
        n_envs=8,
        total_steps=2_000_000,
        pretrained_path=s2_path,
    )

    if s3_path:
        evaluate_model(s3_path, STAGE_3_COINS, label="Stage3_final_eval")
        # 최종 모델을 live.py가 참조하는 경로로 복사
        final_live = os.path.join(ELITE_DIR, "v30_best_model_2h")
        import shutil
        shutil.copy(f"{s3_path}.zip", f"{final_live}.zip")
        print(f"\n  최종 모델  {final_live}.zip 배포 완료!")

    print("\n V30 대함대 훈련 완료. 발진 준비 완료!")


if __name__ == "__main__":
    run_v30_curriculum()
