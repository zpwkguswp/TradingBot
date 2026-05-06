"""
V33 Test Evaluation Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
저장된 v33 모델을 불러와 2026년 이후 데이터(Test Set)에서 성과를 검증합니다.
"""

import os
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from v33_train_full_history import V33_FullHistoryEnv, FULL_UNIVERSE, EVAL_ENV_PARAMS, TCN6LayerExtractor, N_STACK

# 설정
MODEL_PATH = "elite_weights/best_model.zip"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def build_test_venv(coin_list):
    def make_env():
        # 'test' 스플릿 (2026년 이후 데이터 필터링 기능 추가)
        # 여기서는 V33_FullHistoryEnv를 상속받거나 내부 로직을 test용으로 조정
        # V33_FullHistoryEnv는 90/10 split이므로, eval split(마지막 10%)을 사용함
        return V33_FullHistoryEnv(coin_list=coin_list, data_dir="data_storage", split="eval", **EVAL_ENV_PARAMS)
    
    venv = DummyVecEnv([make_env])
    return VecFrameStack(venv, n_stack=N_STACK)

def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    print(f"\n[V33 Evaluation] Loading best model from {MODEL_PATH}...")
    
    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    model = PPO.load(MODEL_PATH, device=DEVICE, custom_objects=custom_objects)
    
    test_venv = build_test_venv(FULL_UNIVERSE)
    
    obs = test_venv.reset()
    done_flags = np.zeros(test_venv.num_envs, dtype=bool)
    
    results = []
    
    print("\n[V33 Evaluation] Running evaluation on 10% tail data...")
    
    # 에피소드 하나가 끝날 때까지 실행
    while not done_flags.all():
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_venv.step(action)
        
        for i, info in enumerate(infos):
            if done_flags[i]: continue
            if dones[i]: done_flags[i] = True
            
            if info.get("trade_closed"):
                results.append({
                    "pnl": info.get("trade_Actual_PnL", 0.0),
                    "reason": info.get("trade_Exit_Reason", ""),
                    "duration": info.get("trade_Holding_Steps", 0)
                })

    test_venv.close()
    
    if not results:
        print("\n[Warning] 평가 기간 중 거래가 발생하지 않았습니다.")
        return

    df = pd.DataFrame(results)
    win_rate = (df['pnl'] > 0).mean() * 100
    avg_pnl = df['pnl'].mean() * 100
    total_pnl = df['pnl'].sum() * 100
    
    pos_pnl = df[df['pnl'] > 0]['pnl'].sum()
    neg_pnl = abs(df[df['pnl'] < 0]['pnl'].sum())
    pf = pos_pnl / neg_pnl if neg_pnl > 0 else float('inf')

    print("\n" + "="*50)
    print("       V33 EVALUATION RESULTS (Tail 10% Data)")
    print("="*50)
    print(f"  총 거래 수      : {len(df)}회")
    print(f"  승률            : {win_rate:.2f}%")
    print(f"  평균 수익률     : {avg_pnl:+.4f}%")
    print(f"  누적 수익률     : {total_pnl:+.2f}%")
    print(f"  Profit Factor   : {pf:.4f}")
    print(f"  평균 보유 기간  : {df['duration'].mean():.1f} steps")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
