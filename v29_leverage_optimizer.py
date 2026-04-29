import pandas as pd
import numpy as np
import os

# ==============================================================================
# [V29 Risk Parity] Leverage Optimizer
# ==============================================================================

TARGET_COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"]
MAX_MDD_LIMIT = 2.5  # 2.5%
LEVERAGE_RANGE = range(1, 21)  # 1x to 20x

def calculate_metrics(pnl_series, leverage):
    equity = 1.0
    equity_curve = [equity]
    
    for pnl in pnl_series:
        # Equity *= (1 + (PnL_% / 100) * leverage)
        equity *= (1 + (pnl / 100.0) * leverage)
        equity_curve.append(equity)
        
    equity_curve = np.array(equity_curve)
    
    # Calculate Total Return (%)
    total_return = (equity - 1.0) * 100.0
    
    # Calculate MDD (%)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100.0
    
    # Calmar Ratio
    calmar = total_return / (max_drawdown + 1e-9)
    
    return total_return, max_drawdown, calmar

def optimize():
    print(f"{'Ticker':<10} | {'Lev':<4} | {'Return (%)':<12} | {'MDD (%)':<10} | {'Calmar':<8}")
    print("-" * 60)
    
    leverage_map = {}
    
    for ticker in TARGET_COINS:
        file_path = f"v29_postmortem_{ticker}.csv"
        if not os.path.exists(file_path):
            print(f" [!] {ticker}: File not found ({file_path}) -> Defaulting to 2x")
            leverage_map[ticker] = 2
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty or 'PnL_%' not in df.columns:
                print(f" [!] {ticker}: No valid data in CSV -> Defaulting to 2x")
                leverage_map[ticker] = 2
                continue
                
            pnl_series = df['PnL_%'].values
            
            best_lev = 2
            best_calmar = -float('inf')
            best_metrics = (0.0, 0.0, 0.0)
            found_safe = False
            
            for lev in LEVERAGE_RANGE:
                ret, mdd, calmar = calculate_metrics(pnl_series, lev)
                
                # Hard Constraint: MDD <= 2.5%
                if mdd <= MAX_MDD_LIMIT:
                    found_safe = True
                    if calmar > best_calmar:
                        best_calmar = calmar
                        best_lev = lev
                        best_metrics = (ret, mdd, calmar)
            
            if not found_safe:
                # Fallback to 2x but calculate metrics for display
                best_lev = 2
                best_metrics = calculate_metrics(pnl_series, best_lev)
                print(f"{ticker:<10} | {best_lev:<4} | {best_metrics[0]:12.2f} | {best_metrics[1]:10.2f} | {best_metrics[2]:8.2f} (REJECTED-FALLBACK)")
            else:
                print(f"{ticker:<10} | {best_lev:<4} | {best_metrics[0]:12.2f} | {best_metrics[1]:10.2f} | {best_metrics[2]:8.2f}")
            
            leverage_map[ticker] = int(best_lev)
            
        except Exception as e:
            print(f" [X] {ticker}: Error processing - {e}")
            leverage_map[ticker] = 2

    print("\n" + "="*60)
    print(" [V29 LEVERAGE_MAP] Copy and paste this into v29_bybit_live.py:")
    print("="*60)
    print("LEVERAGE_MAP = {")
    for ticker, lev in leverage_map.items():
        print(f"    '{ticker}': {lev},")
    print("}")
    print("="*60)

if __name__ == "__main__":
    optimize()
