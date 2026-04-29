import os
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from v29_env import V29_Universal_Env

# ─────────────────────────────────────────────────────────
# [설정] 검증 유니버스 및 모델 정보 (Trial 59 황금 수치)
# ─────────────────────────────────────────────────────────
DATA_DIR = "data_storage"
ELITE_DIR = "elite_weights"
N_STACK = 4

TARGET_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT", "NEARUSDT"]
TIMEFRAME = "2h"
MODEL_PATH = os.path.join(ELITE_DIR, f"v29_best_model_{TIMEFRAME}.zip")

BEST_PARAMS = {
    "far_th": 2.2044178686597258,
    "sl_atr_coef": 3.8350847145336115,
    "trail_act": 0.020,
    "adx_th": 0.169327336793444,      # 물리적 진입 필터 (ADX 최소값)
    "max_disp": 0.0587715969881679    # 물리적 진입 필터 (이격도 최댓값)
}

# ─────────────────────────────────────────────────────────
# [모델 아키텍처] TCN6LayerExtractor (v29_train.py와 동일해야 로드 가능)
# ─────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]

class TCN6LayerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.obs_dim = 36
        self.coin_dim = 1
        self.total_dim = self.obs_dim + self.coin_dim
        
        self.tcn = nn.Sequential(
            CausalConv1d(self.obs_dim, 16, 12, 1), nn.ReLU(), nn.BatchNorm1d(16),
            CausalConv1d(16, 32, 8, 2), nn.ReLU(), nn.BatchNorm1d(32),
            CausalConv1d(32, 64, 5, 4), nn.ReLU(), nn.BatchNorm1d(64),
            CausalConv1d(64, 128, 3, 8), nn.ReLU(), nn.BatchNorm1d(128),
            CausalConv1d(128, 256, 3, 16), nn.ReLU(), nn.BatchNorm1d(256),
            CausalConv1d(256, 256, 2, 32), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.coin_embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.fc = nn.Linear(256 + 16, features_dim)

    def forward(self, observations):
        n_stack = 4
        x_unrolled = observations.view(-1, n_stack, self.total_dim)
        x_features = x_unrolled[:, :, :self.obs_dim].transpose(1, 2)
        tcn_out = self.tcn(x_features)
        coin_idx = x_unrolled[:, 0, self.obs_dim].long()
        emb_out = self.coin_embedding(coin_idx)
        combined = th.cat([tcn_out, emb_out], dim=1)
        return self.fc(combined)

# ─────────────────────────────────────────────────────────
# [함수] 환경 생성 및 테스트 실행
# ─────────────────────────────────────────────────────────
def make_test_env(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}_{TIMEFRAME}.parquet")
    if not os.path.exists(file_path):
        print(f"[!] {ticker} 데이터가 {DATA_DIR}에 없습니다. 스킵.")
        return None

    coin_id = TARGET_COINS.index(ticker)
    def _init():
        return V29_Universal_Env(
            data_dir=DATA_DIR, 
            coin_files=[f"{ticker}_{TIMEFRAME}.parquet"], 
            coin_id=coin_id,
            split_type="eval",
            target_profit=0.008,
            far_th=BEST_PARAMS["far_th"],
            sl_atr_coef=BEST_PARAMS["sl_atr_coef"],
            adx_th=BEST_PARAMS["adx_th"],
            max_disp=BEST_PARAMS["max_disp"],
            trail_act=BEST_PARAMS["trail_act"]
        )
    return _init

def run_v29_oos_test():
    print(f">> [V29 Phase 2] Universal Alpha OOS Cross Validator starting")
    print(f">> Model Path: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("[X] Error: 모델 파일이 없습니다. elite_weights 폴더를 확인하십시오.")
        return

    summary_list = []
    
    # [!] [중요] 모델 로드 시 커스텀 extractor 등록 필수
    custom_objects = {
        "features_extractor_class": TCN6LayerExtractor
    }

    for ticker in TARGET_COINS:
        print(f"\n[*] {ticker} 시뮬레이션 중...", end="", flush=True)
        env_fn = make_test_env(ticker)
        if env_fn is None: continue

        venv = DummyVecEnv([env_fn])
        venv = VecFrameStack(venv, n_stack=N_STACK)
        
        # CPU 검증 (속도 및 안정성)
        model = PPO.load(MODEL_PATH, env=venv, device="cpu", custom_objects=custom_objects)
        
        obs = venv.reset()
        done = False
        trade_logs = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = venv.step(action)
            
            info = infos[0]
            done = done_arr[0]

            if info.get("trade_closed"):
                log_entry = {
                    "Ticker": ticker,
                    "Entry_Price": info.get("trade_Entry_Price"),
                    "Exit_Price":  info.get("trade_Exit_Price"),
                    "PnL_%":       info.get("trade_Actual_PnL", 0) * 100,
                    "MFE_%":       info.get("trade_MFE", 0) * 100,
                    "MAE_%":       info.get("trade_MAE", 0) * 100,
                    "Exit_Reason": info.get("trade_Exit_Reason"),
                    "Entry_ADX":   info.get("trade_entry_adx", float("nan")),
                    "Disparity":   info.get("trade_Entry_Disparity"),
                }
                trade_logs.append(log_entry)

        # 결과 집계 및 저장
        if trade_logs:
            df_log = pd.DataFrame(trade_logs)
            df_log.to_csv(f"v29_postmortem_{ticker}.csv", index=False)
            
            # Profit Factor 계산
            wins = df_log[df_log['PnL_%'] > 0]['PnL_%'].sum()
            losses = abs(df_log[df_log['PnL_%'] < 0]['PnL_%'].sum())
            pf = wins / (losses + 1e-9)
            
            ret_pct = (info.get('bot_equity', 1.0) - 1.0) * 100
            mdd_pct = info.get('bot_mdd', 0.0) * 100
            
            summary_list.append({
                "Ticker": ticker,
                "Trades": len(trade_logs),
                "PF": round(pf, 2),
                "Return(%)": f"{ret_pct:.2f}%",
                "MDD(%)": f"{mdd_pct:.2f}%"
            })
            print(f" 완료! (Trades: {len(trade_logs)}, PF: {pf:.2f})")
        else:
            print(" 매매 없음.")
        
        venv.close()

    # 최종 유니버스 리포트 출력
    print("\n" + "="*60)
    print("--- [V29 Universe Validation Report] ---")
    print("="*60)
    if summary_list:
        df_summary = pd.DataFrame(summary_list)
        print(df_summary.to_string(index=False))
        
        # 평균 성능 계산
        avg_ret = df_summary["Return(%)"].str.replace("%", "").astype(float).mean()
        avg_mdd = df_summary["MDD(%)"].str.replace("%", "").astype(float).mean()
        avg_pf = df_summary["PF"].mean()
        total_trades = df_summary["Trades"].sum()
        
        print("-" * 60)
        print(f"TOTAL: {total_trades} trades | Avg PF: {avg_pf:.2f} | Avg Return: {avg_ret:.2f}% | Avg MDD: {avg_mdd:.2f}%")
        print("=" * 60)
        print("[!] v29_postmortem_[TICKER].csv 파일이 생성되었습니다.")
    else:
        print("검증된 매매 내역이 없습니다. 데이터를 확인하십시오.")

if __name__ == "__main__":
    run_v29_oos_test()
