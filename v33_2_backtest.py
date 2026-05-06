"""
V33_2 백테스트 (Best Model 평가)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 대상  : elite_weights/v33_2_best_model.zip
■ 데이터: 각 코인의 마지막 10% (훈련 중 한 번도 안 본 구간)
■ 출력  : PF, 승률, MDD, 누적수익, 매매횟수, Capture Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import warnings
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

warnings.filterwarnings("ignore")

from v29_env import V29_Universal_Env
from v30_train import FULL_UNIVERSE, TCN6LayerExtractor

# ── 설정 ──────────────────────────────────────────────────
MODEL_PATH  = "elite_weights/v33_2_best_model.zip"
DATA_DIR    = "data_storage"
N_STACK     = 4
N_COINS     = 20          # 평가할 코인 수 (많을수록 정확, 느림)
N_EPISODES  = 3           # 코인당 에피소드 수
MAX_STEPS   = 5000        # 에피소드당 최대 스텝 (5m봉 기준 약 17일)
HOLD_THR    = 0.05        # 진입 신호 임계값

# 평가용 환경 파라미터 (필터 최소화 → 모델이 실제로 진입하게 함)
TEST_ENV_PARAMS = dict(
    target_profit = 0.008,
    far_th        = 10.0,   # 완화
    sl_atr_coef   = 3.835,
    adx_th        = 0.0,    # 완화 (ADX 필터 해제)
    max_disp      = 10.0,   # 완화
    trail_act     = 0.020,
)

FULL_FILE_SUFFIX = "_5m_full.parquet"


# ── 테스트 환경 ────────────────────────────────────────────
class V33_TestEnv(V29_Universal_Env):
    """백테스트 전용 환경: test split (마지막 10%) 사용"""

    def __init__(self, coin_list, data_dir, **kwargs):
        self.data_dir     = data_dir
        self.split        = "test"
        self.full_universe = coin_list

        init_coin = coin_list[0]
        super().__init__(
            data_dir   = data_dir,
            coin_files = [f"{init_coin}{FULL_FILE_SUFFIX}"],
            coin_id    = 0,
            split_type = None,
            **kwargs
        )

        self.adx_th    = TEST_ENV_PARAMS.get("adx_th", 0.169)
        self.max_disp  = TEST_ENV_PARAMS.get("max_disp", 0.058)
        self.far_th    = TEST_ENV_PARAMS.get("far_th", 2.204)
        self.hold_signal_thr = HOLD_THR

        self._ep_step_cnt  = 0
        self._max_ep_steps = MAX_STEPS
        self._trades       = []  # 거래 기록

    def _load_coin_data(self, coin):
        path = os.path.join(self.data_dir, f"{coin}{FULL_FILE_SUFFIX}")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 피처 계산 (V29 호환)
        df["returns"]    = df["close"].pct_change().fillna(0)
        df["log_volume"] = np.log1p(df["volume"])
        df["hl_ratio"]   = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        for s in [5, 10, 20, 60]:
            df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
        for s in [20, 60, 200]:
            df[f"h1_ema_{s}"] = df["close"].ewm(span=s * 12, adjust=False).mean()
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)
        df["atr_raw"] = df["atr_raw"].fillna(0.01).clip(0.001, 0.1)

        # [V33-2] 부모 클래스 하드 필터 중립화
        df["structural_rev_long"] = 1.0
        df["structural_rev_short"] = 1.0
        df["disparity_200"] = 1.0

        # test split: 마지막 10%
        n = len(df)
        cut = int(n * 0.90)
        return df.iloc[cut:].reset_index(drop=True)

    def reset(self, seed=None, options=None):
        self._ep_step_cnt = 0
        coin = np.random.choice(self.full_universe)
        df   = self._load_coin_data(coin)
        if df is None or len(df) < MAX_STEPS + 100:
            coin = self.full_universe[0]
            df   = self._load_coin_data(coin)
        self.df       = df
        self.coin_id  = self.full_universe.index(coin) if coin in self.full_universe else 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # Desaturation
        if isinstance(action, np.ndarray):
            raw    = np.clip(action.astype(np.float64), -0.9999, 0.9999)
            action = np.tanh(np.arctanh(raw) * 0.25).astype(np.float32)

        was_in = getattr(self, 'position', None) is not None
        obs, reward, done, truncated, info = super().step(action)
        self._ep_step_cnt += 1
        if self._ep_step_cnt >= self._max_ep_steps:
            done = True

        now_in = getattr(self, 'position', None) is not None
        if was_in and not now_in:
            init  = float(getattr(self, 'initial_balance',
                          getattr(self, '_initial_balance_ref', 10_000.0)))
            if init <= 0: init = 10_000.0
            eq    = float(getattr(self, 'balance', init)) / init
            self._trades.append({
                "pnl":    float(info.get("trade_Actual_PnL", 0.0)),
                "reason": str(info.get("trade_Exit_Reason", "")),
                "coin":   int(getattr(self, 'coin_id', -1)),
                "equity": round(eq, 6),
            })

        return obs, reward, done, truncated, info


# ── 지표 계산 ──────────────────────────────────────────────
def compute_metrics(trades):
    if not trades:
        return None
    df    = pd.DataFrame(trades)
    total = len(df)
    wins  = df[df["pnl"] > 0]
    loss  = df[df["pnl"] < 0]
    wr    = len(wins) / total * 100
    gp    = wins["pnl"].sum() * 100
    gl    = abs(loss["pnl"].sum()) * 100
    net   = gp - gl
    pf    = gp / gl if gl > 0 else float("inf")

    # Peak equity → MDD
    eq_series = df["equity"].values
    peak, mdd = eq_series[0], 0.0
    for e in eq_series:
        if e > peak: peak = e
        dd = (peak - e) / (peak + 1e-9)
        if dd > mdd: mdd = dd

    exit_counts = df["reason"].value_counts().to_dict()
    return dict(total=total, wr=wr, gp=gp, gl=gl, net=net,
                pf=pf, mdd=mdd * 100, exit=exit_counts,
                best=df["pnl"].max() * 100,
                worst=df["pnl"].min() * 100)


# ── 메인 ──────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  [V33_2 백테스트] best_model → 테스트 데이터 평가")
    print("=" * 64)

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ 모델 없음: {MODEL_PATH}")
        return

    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
    print(f"  ✅ 모델 로드: {MODEL_PATH}")

    coins = FULL_UNIVERSE[:N_COINS]
    print(f"  📊 평가 코인: {N_COINS}개 × {N_EPISODES}에피소드 × {MAX_STEPS}스텝\n")

    all_trades = []
    for ep_idx in range(N_EPISODES):
        # 매 에피소드마다 새 환경
        def _make():
            return V33_TestEnv(coin_list=FULL_UNIVERSE, data_dir=DATA_DIR, **TEST_ENV_PARAMS)
        venv = VecFrameStack(DummyVecEnv([_make]), n_stack=N_STACK)

        obs = venv.reset()
        done_flags = [False]
        env_ref = venv.envs[0]  # DummyVecEnv 내부 접근

        while not all(done_flags):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_flags, _ = venv.step(action)

        trades_this = env_ref._trades
        all_trades.extend(trades_this)
        venv.close()

        n = len(trades_this)
        print(f"  Episode {ep_idx+1}/{N_EPISODES}: {n}건 거래")

    # 결과 출력
    m = compute_metrics(all_trades)
    if not m:
        print("\n  ⚠️  거래 없음 - 모델이 HOLD만 하고 있습니다.")
        return

    pf_s = f"{m['pf']:.4f}" if m['pf'] != float('inf') else "∞"
    print(f"\n{'='*64}")
    print(f"  ── 백테스트 결과 (테스트 데이터, 미래 미포함) ─────────────")
    print(f"{'='*64}")
    print(f"  {'매매횟수':<14} {m['total']:>8,} 회")
    print(f"  {'승률':<14} {m['wr']:>8.2f} %")
    print(f"  {'PF':<14} {pf_s:>8}")
    print(f"  {'MDD':<14} {m['mdd']:>8.2f} %")
    print(f"  {'누적 수익률':<13} {m['net']:>+8.2f} %")
    print(f"  {'최대 단일수익':<12} {m['best']:>+8.2f} %")
    print(f"  {'최대 단일손실':<12} {m['worst']:>+8.2f} %")
    print(f"  {'수익/MDD':<14} {m['net']/(m['mdd']+1e-9):>8.4f}")
    print(f"\n  [청산 사유]")
    for reason, cnt in m['exit'].items():
        print(f"    {reason:<12} {cnt:>5,} 회  ({cnt/m['total']*100:.1f}%)")
    print(f"\n{'='*64}")

    # 종합 판단
    print(f"\n  📋 종합 판단:")
    if m['pf'] > 1.5 and m['wr'] > 45 and m['mdd'] < 15:
        print(f"  ✅ 우수 - 라이브 배포 검토 가능")
    elif m['pf'] > 1.2 and m['wr'] > 40:
        print(f"  🟡 보통 - 추가 훈련 후 재평가 권장")
    else:
        print(f"  ❌ 미흡 - 훈련 계속 필요")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
