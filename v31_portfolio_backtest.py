import numpy as np
import pandas as pd
import itertools
import multiprocessing
import os
import torch
from functools import partial
from time import time
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

# Import necessary components from the actual live/training engine
from v30_train import TCN6LayerExtractor, FULL_UNIVERSE
from v29_env import V29_Universal_Env

def load_v30_model(model_path):
    print(f"Loading V30 Model from {model_path}...")
    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
    try:
        model = PPO.load(model_path, device="cpu", custom_objects=custom_objects)
        print("[SUCCESS] Model loaded successfully (Standard).")
        return model
    except ValueError as e:
        if "parameter groups" in str(e):
            print("[INFO] Optimizer mismatch detected. Loading weights only...")
            
            # Dummy environment to initialize model architecture
            import gymnasium as gym
            class DummyEnv(gym.Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(148,), dtype=np.float32)
                    self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                def reset(self, seed=None, options=None): return np.zeros(148, dtype=np.float32), {}
                def step(self, action): return np.zeros(148, dtype=np.float32), 0.0, False, False, {}
            
            policy_kwargs = dict(
                features_extractor_class=TCN6LayerExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[128, 64], vf=[256, 128]),
            )
            model = PPO("MlpPolicy", DummyVecEnv([lambda: DummyEnv()]), 
                             policy_kwargs=policy_kwargs, device="cpu")
            
            import zipfile
            import io
            with zipfile.ZipFile(model_path, "r") as archive:
                with archive.open("policy.pth") as f:
                    policy_weights = torch.load(io.BytesIO(f.read()), map_location="cpu")
            model.policy.load_state_dict(policy_weights)
            print("[SUCCESS] Model loaded successfully (Weights Only).")
            return model
        else:
            raise e

def generate_real_data(num_candles=1000):
    """
    Generate real price and act_val data by running historical data through the model.
    """
    print("Generating real data using V30 model...")
    model_path = "elite_weights/v30_best_model_2h.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please check the path.")
        
    model = load_v30_model(model_path)
    
    all_prices = {}
    all_act_vals = {}
    
    for coin_idx, ticker in enumerate(FULL_UNIVERSE):
        print(f"Processing {ticker} ({coin_idx+1}/{len(FULL_UNIVERSE)})...")
        
        try:
            # Initialize env for this specific coin
            env = V29_Universal_Env(
                data_dir="data_storage",
                coin_files=[f"{ticker}_2h.parquet"],
                coin_id=coin_idx,
                split_type=None,
                target_profit=0.008,
                far_th=2.2044178686597258,
                sl_atr_coef=3.8350847145336115,
                adx_th=0.169327336793444,
                max_disp=0.0587715969881679,
                trail_act=0.020
            )
            
            raw_obs = env.reset()
            if isinstance(raw_obs, tuple): raw_obs = raw_obs[0]
            
            prices_history = []
            obs_history = [raw_obs]
            
            done = False
            while not done:
                # scale defense to prevent bankruptcy in env
                cur_bal = getattr(env, 'balance', 10000.0)
                cur_nw = getattr(env, 'net_worth', 10000.0)
                if cur_bal <= 0 or cur_nw <= 0:
                    if hasattr(env, 'balance'): env.balance = 10000.0
                    if hasattr(env, 'net_worth'): env.net_worth = 10000.0
                    if hasattr(env, 'max_net_worth'): env.max_net_worth = 10000.0
                    if hasattr(env, 'initial_balance'): env.initial_balance = 10000.0
                    
                res = env.step(np.array([0.0]))
                raw_obs = res[0]
                done = res[2] if len(res) == 4 else (res[2] or res[3])
                
                obs_history.append(raw_obs)
                prices_history.append(env.df.iloc[env.current_step]['close'])
                
            # We want the last `num_candles` steps
            prices_history = prices_history[-num_candles:]
            
            # Extract 4-frame stacked observations
            stacked_obs_list = []
            start_idx = len(obs_history) - num_candles
            
            for t in range(num_candles):
                idx = start_idx + t
                # ensure we don't go out of bounds
                idx = max(3, idx)
                
                last_4_raw = obs_history[idx-3:idx+1]
                frames = []
                for obs in last_4_raw:
                    new_obs = np.zeros(37, dtype=np.float32)
                    obs_1d = np.nan_to_num(np.array(obs).flatten())
                    copy_len = min(len(obs_1d), 36)
                    new_obs[:copy_len] = obs_1d[:copy_len]
                    new_obs[36] = float(coin_idx)
                    frames.append(new_obs)
                
                stacked_obs = np.concatenate(frames)
                stacked_obs_list.append(stacked_obs)
                
            # Batch predict
            batch_obs = np.array(stacked_obs_list)
            actions, _ = model.predict(batch_obs, deterministic=True)
            
            all_prices[ticker] = prices_history
            all_act_vals[ticker] = actions.flatten()
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            all_prices[ticker] = [np.nan] * num_candles
            all_act_vals[ticker] = [0.0] * num_candles

    df_prices = pd.DataFrame(all_prices)
    df_act_vals = pd.DataFrame(all_act_vals)
    
    # Fill any NaNs
    df_prices = df_prices.ffill().bfill()
    df_act_vals = df_act_vals.fillna(0.0)
    
    print(f"Real data generated: {len(FULL_UNIVERSE)} coins, {num_candles} candles.")
    return df_prices, df_act_vals

def run_simulation(params, prices_data, act_vals_data):
    """
    Run a single backtest simulation with the given parameters.
    """
    (max_positions, allocation_rate, entry_cutoff, 
     swap_threshold_ratio, swap_min_diff, swap_cooldown) = params

    initial_capital = 10000.0
    capital = initial_capital
    positions = {}  
    
    peak_equity = initial_capital
    mdd = 0.0
    
    num_candles = len(prices_data)
    fee_rate = 0.0005
    
    trades_count = 0
    wins_count = 0
    
    for t in range(num_candles):
        current_prices = prices_data[t]
        current_act_vals = act_vals_data[t]
        
        # 1. Update positions state, calculate PnL, and check Exits (SL, Trailing, Flip)
        for coin in list(positions.keys()):
            pos = positions[coin]
            curr_price = current_prices[coin]
            act_val_now = current_act_vals[coin]
            
            # Update act_val for current positions to the latest absolute value
            pos['act_val'] = abs(act_val_now)
            
            # Calculate raw PnL based on position side
            if pos['side'] == 'long':
                pnl_raw = (curr_price / pos['entry_price']) - 1.0
                current_value = pos['margin'] * (curr_price / pos['entry_price'])
            else:
                pnl_raw = 1.0 - (curr_price / pos['entry_price'])
                current_value = pos['margin'] * (1.0 + pnl_raw)
                
            # Track Max Favorable Excursion (MFE)
            if pnl_raw > pos['mfe']:
                pos['mfe'] = pnl_raw
                
            # Exit Rules Check
            # Rule 1: Signal Flip
            signal_flip = False
            if pos['side'] == 'long' and act_val_now < -0.05:
                signal_flip = True
            elif pos['side'] == 'short' and act_val_now > 0.05:
                signal_flip = True
                
            # Rule 2: SL and Simplified Trailing Stop
            sl_hit = (pnl_raw <= -0.03)
            trail_hit = (pos['mfe'] >= 0.02 and pnl_raw < pos['mfe'] * 0.70)
            
            if signal_flip or sl_hit or trail_hit:
                # Liquidate
                notional_value = pos['amount'] * curr_price
                sell_fee = notional_value * fee_rate
                trade_exit_value = current_value - sell_fee
                
                # Stats
                trades_count += 1
                if trade_exit_value > pos['invest_amount']:
                    wins_count += 1
                    
                capital += trade_exit_value
                del positions[coin]
                
        # 2. Calculate current total equity (Mark to Market)
        total_equity = capital
        for coin, pos in positions.items():
            curr_price = current_prices[coin]
            if pos['side'] == 'long':
                current_value = pos['margin'] * (curr_price / pos['entry_price'])
            else:
                pnl_raw = 1.0 - (curr_price / pos['entry_price'])
                current_value = pos['margin'] * (1.0 + pnl_raw)
            total_equity += current_value
            
        # Update MDD
        if total_equity > peak_equity:
            peak_equity = total_equity
        drawdown = (peak_equity - total_equity) / peak_equity
        if drawdown > mdd:
            mdd = drawdown
            
        # 3. Filter and rank candidates (abs(act_val) >= ENTRY_CUTOFF)
        valid_mask = np.abs(current_act_vals) >= entry_cutoff
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            continue
            
        # Sort valid indices by abs(act_val) descending
        valid_act_vals = np.abs(current_act_vals[valid_indices])
        sorted_order = np.argsort(-valid_act_vals)
        ranked_candidates = valid_indices[sorted_order]
        
        for cand_idx in ranked_candidates:
            if cand_idx in positions:
                continue # Already in a position with this coin
                
            cand_act_val_raw = current_act_vals[cand_idx]
            cand_act_val = abs(cand_act_val_raw)
            cand_price = current_prices[cand_idx]
            cand_side = 'long' if cand_act_val_raw > 0 else 'short'
            
            if len(positions) < max_positions:
                # Enter new position
                invest_amount = total_equity * allocation_rate
                
                # Ensure we don't spend more cash than we have
                if invest_amount > capital:
                    invest_amount = capital
                
                if invest_amount <= 0:
                    continue
                    
                fee = invest_amount * fee_rate
                margin = invest_amount - fee
                amount = margin / cand_price
                capital -= invest_amount
                
                positions[cand_idx] = {
                    'side': cand_side,
                    'entry_price': cand_price,
                    'amount': amount,
                    'margin': margin,
                    'invest_amount': invest_amount,
                    'entry_time': t,
                    'act_val': cand_act_val,
                    'mfe': 0.0
                }
            else:
                # Swap logic: Find the weakest position
                weakest_coin = min(positions.keys(), key=lambda c: positions[c]['act_val'])
                weakest_pos = positions[weakest_coin]
                weakest_act_val = weakest_pos['act_val']
                
                # Rule 3: 1.0 Saturation Swap Treatment (Tie-breaker)
                if cand_act_val >= 0.95 and weakest_act_val >= 0.95:
                    break
                
                # Swap Conditions
                cond1 = cand_act_val > weakest_act_val * (1 + swap_threshold_ratio)
                cond2 = (cand_act_val - weakest_act_val) > swap_min_diff
                cond3 = (t - weakest_pos['entry_time']) > swap_cooldown
                
                if cond1 and cond2 and cond3:
                    # Liquidate weakest
                    weak_price = current_prices[weakest_coin]
                    if weakest_pos['side'] == 'long':
                        weak_pnl = (weak_price / weakest_pos['entry_price']) - 1.0
                        weak_value = weakest_pos['margin'] * (weak_price / weakest_pos['entry_price'])
                    else:
                        weak_pnl = 1.0 - (weak_price / weakest_pos['entry_price'])
                        weak_value = weakest_pos['margin'] * (1.0 + weak_pnl)
                        
                    sell_fee = (weakest_pos['amount'] * weak_price) * fee_rate
                    trade_exit_value = weak_value - sell_fee
                    
                    # Stats
                    trades_count += 1
                    if trade_exit_value > weakest_pos['invest_amount']:
                        wins_count += 1
                        
                    capital += trade_exit_value
                    del positions[weakest_coin]
                    
                    # Enter new position
                    invest_amount = total_equity * allocation_rate
                    if invest_amount > capital:
                        invest_amount = capital
                    
                    if invest_amount > 0:
                        fee = invest_amount * fee_rate
                        margin = invest_amount - fee
                        amount = margin / cand_price
                        capital -= invest_amount
                        
                        positions[cand_idx] = {
                            'side': cand_side,
                            'entry_price': cand_price,
                            'amount': amount,
                            'margin': margin,
                            'invest_amount': invest_amount,
                            'entry_time': t,
                            'act_val': cand_act_val,
                            'mfe': 0.0
                        }
                else:
                    break
                    
    # Final liquidation at the end of the simulation
    final_equity = capital
    for coin, pos in positions.items():
        curr_price = prices_data[-1][coin]
        if pos['side'] == 'long':
            pnl_raw = (curr_price / pos['entry_price']) - 1.0
            current_value = pos['margin'] * (curr_price / pos['entry_price'])
        else:
            pnl_raw = 1.0 - (curr_price / pos['entry_price'])
            current_value = pos['margin'] * (1.0 + pnl_raw)
            
        sell_fee = (pos['amount'] * curr_price) * fee_rate
        trade_exit_value = current_value - sell_fee
        
        # Stats
        trades_count += 1
        if trade_exit_value > pos['invest_amount']:
            wins_count += 1
            
        final_equity += trade_exit_value
        
    net_pnl = (final_equity - initial_capital) / initial_capital
    win_rate = (wins_count / trades_count) if trades_count > 0 else 0.0
    
    return {
        'MAX_POSITIONS': max_positions,
        'ALLOCATION_RATE': allocation_rate,
        'ENTRY_CUTOFF': entry_cutoff,
        'SWAP_THRESHOLD_RATIO': swap_threshold_ratio,
        'SWAP_MIN_DIFF': swap_min_diff,
        'SWAP_COOLDOWN': swap_cooldown,
        'Net_PnL': net_pnl,
        'MDD': mdd,
        'Total_Trades': trades_count,
        'Win_Rate': win_rate
    }

def main():
    print("=== V31 Portfolio Grid Search Backtester ===")
    
    # 1. Generate Real Data from Environment & Model
    df_prices, df_act_vals = generate_real_data(num_candles=1000)
    
    # Convert DataFrames to numpy arrays for faster access during simulation loop
    prices_data = df_prices.values
    act_vals_data = df_act_vals.values
    
    # 2. Define Grid Search Space (Extreme Sniper Mode)
    grid_params = {
        'MAX_POSITIONS': [1, 2],
        'ALLOCATION_RATE': [0.10, 0.15],
        'ENTRY_CUTOFF': [0.85, 0.90, 0.95],
        'SWAP_THRESHOLD_RATIO': [0.15, 0.20, 0.25],
        'SWAP_MIN_DIFF': [0.20, 0.30],
        'SWAP_COOLDOWN': [2, 4, 6]
    }
    
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*(grid_params[k] for k in keys)))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    start_time = time()
    
    num_cores = multiprocessing.cpu_count()
    print(f"Running Grid Search on {num_cores} CPU cores...")
    
    # 3. Multiprocessing Pool
    # We use functools.partial to cleanly pass the shared dataset arrays to the worker task
    task = partial(run_simulation, prices_data=prices_data, act_vals_data=act_vals_data)
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(task, combinations)
        
    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.\n")
    
    # 4. Process and Format Results
    df_results = pd.DataFrame(results)
    
    # Sort by Net PnL (descending), then by MDD (ascending)
    df_results = df_results.sort_values(by=['Net_PnL', 'MDD'], ascending=[False, True]).reset_index(drop=True)
    
    # Format columns for display
    df_results_styled = df_results.copy()
    df_results_styled['Net_PnL'] = df_results_styled['Net_PnL'].apply(lambda x: f"{x*100:.2f}%")
    df_results_styled['MDD'] = df_results_styled['MDD'].apply(lambda x: f"{x*100:.2f}%")
    df_results_styled['Win_Rate'] = df_results_styled['Win_Rate'].apply(lambda x: f"{x*100:.2f}%")
    
    print("=== Top 10 Parameter Combinations ===")
    print(df_results_styled.head(10).to_string(index=False))

if __name__ == '__main__':
    # Required for Windows multiprocessing safety
    multiprocessing.freeze_support()
    main()
