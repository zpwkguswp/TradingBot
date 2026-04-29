import numpy as np
import pandas as pd
from gymnasium import spaces
from v28_env import V28_MFE_Env

class V29_Universal_Env(V28_MFE_Env):
    def __init__(self, data_dir: str, coin_files: list, coin_id: int, split_type="train",
                 target_profit=0.008, far_th=2.5, sl_atr_coef=3.2, adx_th=0.0, max_disp=1.0, trail_act=0.0):
        # Use kwargs explicitly to avoid clashing with parent positional args
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

    def _get_obs(self) -> np.ndarray:
        # Get base 36 dims from V28 logic
        obs = super()._get_obs()
        
        row = self.df.iloc[self.current_step]
        close = float(row["close"])
        h1_atr_abs = float(row.get("h1_atr_abs", 0.0001))
        if pd.isna(h1_atr_abs) or h1_atr_abs <= 0: h1_atr_abs = 0.0001
        
        # Override V26 local disparity features (indices 27, 28)
        # obs[27] corresponds to disparity_200. We replace it with disparity_200_atr.
        obs[27] = float(row.get("disparity_200_atr", 0.0))
        # obs[28] was v26_disparity. We replace it with disparity_60_atr.
        obs[28] = float(row.get("disparity_60_atr", 0.0))
        
        # Override V26_3 MTF features at indices 29, 30, 31 
        # In V26_3, base_obs is 29 dims. So h1_features start at index 29.
        obs[29] = (close - float(row.get("h1_ema_20", close))) / h1_atr_abs
        obs[30] = (close - float(row.get("h1_ema_60", close))) / h1_atr_abs
        obs[31] = (close - float(row.get("h1_ema_200", close))) / h1_atr_abs
        
        # Append coin_id (37th dimension)
        final_obs = np.append(obs, float(self.coin_id))
        return final_obs.astype(np.float32)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        atr_pct = float(self.df.iloc[self.current_step].get("atr_raw", 0.01))
        if pd.isna(atr_pct) or atr_pct <= 0: atr_pct = 0.01
        
        # [V29] Reward Shaping
        # Normalizing around 1.0% avg volatility
        norm_factor = atr_pct * 100.0 
        
        scaled_reward = float(reward) / norm_factor
        
        return obs, scaled_reward, done, truncated, info

    def _close_position(self) -> tuple:
        reward, won, mscl_penalty, net_pnl = super()._close_position()
        
        atr_pct = float(self.df.iloc[self.current_step].get("atr_raw", 0.01))
        if pd.isna(atr_pct) or atr_pct <= 0: atr_pct = 0.01
        
        norm_factor = atr_pct * 100.0
        scaled_reward = float(reward) / norm_factor
        
        return (scaled_reward, won, mscl_penalty, net_pnl)
