# -*- coding: utf-8 -*-
"""
V29 Direct Stack 실시간 검증기 (Bybit Real-time Data)
----------------------------------------------------
봇 가동 전, 환경과 데이터가 정상인지 교차 검증하는 독립 테스트 파일.
[핵심 원칙] 정속 주행 풀스캔: Warp(순간이동) 없이 step()을 처음부터 끝까지
밟아 환경 내부 지표 버퍼를 완전히 채운 뒤, 마지막 4 프레임을 추출합니다.
"""
import asyncio
import os
import numpy as np
from stable_baselines3 import PPO

from v29_bybit_live import TCN6LayerExtractor, ticker_to_symbol
from v29_env import V29_Universal_Env
from v29_data_builder import run_builder
from exchange import ExchangeClient
from config import BYBIT_API_KEY, BYBIT_API_SECRET

# ==============================================================================
# 설정
# ==============================================================================
DATA_DIR   = "data_storage"
TIMEFRAME  = "2h"
ELITE_DIR  = "elite_weights"
MODEL_PATH = os.path.join(ELITE_DIR, f"v29_best_model_{TIMEFRAME}.zip")

VERIFY_COINS  = ["BTCUSDT", "ETHUSDT", "NEARUSDT"]
FULL_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT", "NEARUSDT"
]

TARGET_PROFIT = 0.008
FAR_TH        = 2.2044178686597258
SL_ATR_COEF   = 3.8350847145336115
ADX_TH        = 0.169327336793444
MAX_DISP      = 0.0587715969881679
TRAIL_ACT     = 0.020
# ==============================================================================


def fullscan_observation(ticker: str, coin_id: int) -> tuple:
    """
    [정속 주행 풀스캔] Warp 없이 0번 step부터 끝까지 밟아
    환경 내부 버퍼를 실제 데이터로 채운 뒤 마지막 4 프레임을 반환합니다.
    Returns: (stacked_obs (1,148), frames[-1] (37,))
    """
    env = V29_Universal_Env(
        data_dir=DATA_DIR,
        coin_files=[f"{ticker}_{TIMEFRAME}.parquet"],
        coin_id=coin_id,
        split_type=None,
        target_profit=TARGET_PROFIT,
        far_th=FAR_TH,
        sl_atr_coef=SL_ATR_COEF,
        adx_th=ADX_TH,
        max_disp=MAX_DISP,
        trail_act=TRAIL_ACT,
    )

    # 1. 초기화 및 첫 프레임 수집
    reset_result = env.reset()
    raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    max_idx = len(env.df) - 1
    obs_history = [raw_obs]

    # 2. [정속 주행 풀스캔] 0번부터 끝까지 step()을 직접 밟습니다.
    print(f"  [Scan] {ticker}: {max_idx} steps 풀스캔 시작...", flush=True)
    for step_i in range(max_idx):
        # God Mode: 파산으로 인한 내부 에러 방지 (무한 자금 주입)
        env.balance = 1e9
        env.net_worth = 1e9
        if hasattr(env, 'max_net_worth'):
            env.max_net_worth = 1e9

        res = env.step(np.array([0.0]))
        obs_history.append(res[0])

        if (step_i + 1) % 200 == 0:
            print(f"  [Scan] {ticker}: {step_i+1}/{max_idx} 완료...", flush=True)

    print(f"  [Scan] {ticker}: 풀스캔 완료!", flush=True)

    # 3. 마지막 4 프레임 추출
    last_4_raw = obs_history[-4:]

    # 4. 37차원 규격 수동 조립
    frames = []
    for obs in last_4_raw:
        new_obs = np.zeros(37, dtype=np.float32)
        obs_1d = np.nan_to_num(np.array(obs).flatten())
        copy_len = min(len(obs_1d), 36)
        new_obs[:copy_len] = obs_1d[:copy_len]
        new_obs[36] = float(coin_id)
        frames.append(new_obs)

    stacked_obs = np.concatenate(frames).reshape(1, -1)
    return stacked_obs, frames[-1].copy()


async def verify_realtime_obs():
    print("=" * 60)
    print("[V29] Direct Stack 실시간 검증기 시작")
    print("=" * 60)

    # 1. 실시간 데이터 동기화
    print("\n[1단계] 바이비트 실시간 데이터 동기화 및 2H 파케이 빌드...")
    exchange = ExchangeClient(BYBIT_API_KEY, BYBIT_API_SECRET)
    try:
        await exchange.exchange.load_time_difference()
        print("   OK: 서버 시간 동기화 완료")
    except Exception as e:
        print(f"   WARN: 시간 동기화 실패 (계속 진행): {e}")

    for ticker in VERIFY_COINS:
        print(f"   Building {ticker}...")
        try:
            run_builder(ticker)
        except Exception as e:
            print(f"   FAIL: {ticker} 빌드 실패: {e}")

    # 2. 모델 로드
    print(f"\n[2단계] V29 모델 로드: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"   FAIL: 모델 파일 없음! 경로를 확인하십시오: {MODEL_PATH}")
        return

    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
    print("   OK: 모델 로드 완료")

    # 3. 코인별 검증
    print("\n[3단계] Direct Stack 관측값 추출 및 추론 테스트")

    results = []
    for ticker in VERIFY_COINS:
        print(f"\n{'-'*55}")
        print(f"  [{ticker}] 검증 시작")
        try:
            parquet_path = os.path.join(DATA_DIR, f"{ticker}_{TIMEFRAME}.parquet")
            if not os.path.exists(parquet_path):
                print(f"  FAIL: 파케이 파일 없음: {parquet_path}")
                continue

            coin_id = FULL_UNIVERSE.index(ticker)
            stacked_obs, single_obs = fullscan_observation(ticker, coin_id)

            action, _ = model.predict(stacked_obs, deterministic=True)
            act_val = float(action[0])

            signal     = "LONG" if act_val > 0.05 else ("SHORT" if act_val < -0.05 else "WAIT")
            is_warp    = abs(abs(act_val) - 1.0) < 0.001 or abs(abs(act_val) - 0.7907) < 0.005
            ghost_flag = "[GHOST DETECTED]" if is_warp else "[OK]"
            shape_ok   = stacked_obs.shape == (1, 148)

            print(f"  Coin ID        : {coin_id}")
            print(f"  Stack Shape    : {stacked_obs.shape}  (OK=(1, 148))")
            print(f"  Last 5 values  : {single_obs[:5]}")
            print(f"  Prediction     : {act_val:.4f}  ->  {signal}")
            print(f"  Ghost Check    : {ghost_flag}")

            results.append({
                "ticker":    ticker,
                "act_val":   act_val,
                "signal":    signal,
                "saturated": is_warp,
                "shape_ok":  shape_ok,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 4. 최종 요약
    print(f"\n{'=' * 55}")
    print("[최종 검증 요약]")
    all_ok = True
    for r in results:
        shape_flag = "OK" if r["shape_ok"] else "FAIL"
        ghost_flag = "GHOST" if r["saturated"] else "OK"
        print(f"  {r['ticker']:12s} | Shape:{shape_flag} | Ghost:{ghost_flag} | {r['act_val']:.4f} ({r['signal']})")
        if not r["shape_ok"] or r["saturated"]:
            all_ok = False

    if all_ok:
        print("\n[PASS] 전 코인 검증 통과! 메인 봇을 발진시키십시오.")
    else:
        print("\n[FAIL] 검증 실패 항목 존재. 위 경고를 확인 후 봇을 가동하십시오.")
    print("=" * 55)

    try:
        await exchange.exchange.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(verify_realtime_obs())
