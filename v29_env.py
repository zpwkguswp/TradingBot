import numpy as np
import pandas as pd
import os
from gymnasium import spaces
from v28_env import V28_MFE_Env

class V29_Universal_Env(V28_MFE_Env):
    def __init__(self, data_dir: str, coin_files: list, coin_id: int, split_type="train",
                 target_profit=0.008, far_th=2.5, sl_atr_coef=3.2, adx_th=0.0, max_disp=1.0, trail_act=0.020):
        # trail_act 기본값을 실전 규격인 0.020으로 고정
        super().__init__(
            data_dir=data_dir, coin_files=coin_files, split_type=split_type, 
            target_profit=target_profit, far_th=far_th, sl_atr_coef=sl_atr_coef, 
            adx_th=adx_th, max_disp=max_disp, trail_act=trail_act
        )
        
        self.coin_id = coin_id
        
        # Override observation space: 36 (base) + 1 (coin_id) = 37
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )

    def _load_coin_data(self, coin_file):
        """[V34 핵심] 부모의 split 로직을 무시하고 데이터를 100% 로드 (전수 학습)"""
        path = os.path.join(self.data_dir, coin_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # 최근 2026년 데이터까지 포함한 전체 데이터프레임 반환
        return df

    def _get_obs(self) -> np.ndarray:
        # Get base 36 dims from V28 logic
        obs = super()._get_obs()
        
        row = self.df.iloc[self.current_step]
        close = float(row["close"])
        h1_atr_abs = float(row.get("h1_atr_abs", 0.0001))
        if pd.isna(h1_atr_abs) or h1_atr_abs <= 0: h1_atr_abs = 0.0001
        
        # MTF features sync (indices 29, 30, 31)
        obs[29] = (close - float(row.get("h1_ema_20", close))) / h1_atr_abs
        obs[30] = (close - float(row.get("h1_ema_60", close))) / h1_atr_abs
        obs[31] = (close - float(row.get("h1_ema_200", close))) / h1_atr_abs
        
        # Append coin_id (37th dimension)
        final_obs = np.append(obs, float(self.coin_id))
        return final_obs.astype(np.float32)

    def step(self, action):
        # 1. 부모의 step 실행 (V28의 진입 필터 및 MFE 추적 포함)
        obs, reward, done, truncated, info = super().step(action)
        
        # 코인 이름 정보 추가 (로그용)
        info["coin"] = self.current_coin.split('_')[0] if hasattr(self, "current_coin") else "UNKNOWN"
        
        # 2. [V34 핵심] 실전형 트레일링 스탑 강제 집행 (Mirroring Live v1.2)
        if self.position is not None and not done:
            close_now = float(self.df.iloc[self.current_step]["close"])
            entry_p = self.position["entry_price"]
            side = self.position["dir"]
            # 현재 수익률
            pnl = (close_now / entry_p - 1.0) if side == "long" else (1.0 - close_now / entry_p)
            
            # V28_MFE_Env가 이미 mfe를 실시간으로 추적하고 있으므로 활용
            if self.mfe >= self.trail_act: # 2.0% 수익권 진입 시
                # 고점 대비 30% 반납 시 (수익의 70% 보존)
                if pnl < self.mfe * 0.7:
                    # 즉시 청산 및 V29 규격 보상 정산 (n은 net_pnl)
                    r, w, m, n = self._close_position()
                    reward += r
                    
                    # [V34 Fix] 정보 기록 누락 해결 (로그 생성을 위해 필수)
                    info["trade_closed"] = True
                    info["trade_Actual_PnL"] = n
                    info["trade_Exit_Reason"] = "Trailing_Stop_Mirror"
                    info["trade_MFE"] = self.mfe
                    
                    done = True
        
        # 3. 보상 정규화 (V29 고유 로직: ATR 비례 스케일링)
        atr_pct = float(self.df.iloc[self.current_step].get("atr_raw", 0.01))
        if pd.isna(atr_pct) or atr_pct <= 0: atr_pct = 0.01
        norm_factor = atr_pct * 100.0 
        scaled_reward = float(reward) / norm_factor
        
        return obs, scaled_reward, done, truncated, info

    def _close_position(self) -> tuple:
        # 부모의 청산 로직 (Regret Penalty 등) 실행
        reward, won, mscl_penalty, net_pnl = super()._close_position()
        
        # 보상 정규화 적용
        atr_pct = float(self.df.iloc[self.current_step].get("atr_raw", 0.01))
        if pd.isna(atr_pct) or atr_pct <= 0: atr_pct = 0.01
        norm_factor = atr_pct * 100.0
        scaled_reward = float(reward) / norm_factor
        
        return (scaled_reward, won, mscl_penalty, net_pnl)
