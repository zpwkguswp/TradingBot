import requests
import pandas as pd
import os
import time
from datetime import datetime

TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT", "NEARUSDT"]

def download_oos(symbol):
    file_path = f"data_storage/{symbol}_5m.parquet"
    if os.path.exists(file_path):
        print(f"[{symbol}] File already exists. Skipping download.")
        return
        
    start_dt = datetime(2026, 1, 1, 0, 0, 0)
    end_dt = datetime(2026, 4, 16, 23, 59, 59)
    
    since = int(start_dt.timestamp() * 1000)
    final_end_ts = int(end_dt.timestamp() * 1000)
    
    os.makedirs('data_storage', exist_ok=True)
    all_data = []
    
    url = "https://api.bybit.com/v5/market/kline"
    print(f"[{symbol}] Downloading OOS data (2026-01-01 ~ 2026-04-16)...")
    
    while True:
        end_time = since + (1000 * 5 * 60 * 1000) # 1000 candles * 5 minutes * 60 sec * 1000 ms
        if end_time > final_end_ts: end_time = final_end_ts

        params = {"category": "linear", "symbol": symbol, "interval": "5", "start": since, "end": end_time, "limit": 1000}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            res_json = resp.json()
            klines = res_json.get("result", {}).get("list", [])
            
            if not klines:
                if since >= final_end_ts: break
                else:
                    since += (5 * 60 * 1000)
                    continue
            
            df_temp = pd.DataFrame(klines).iloc[:, 0:6]
            df_temp.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_temp = df_temp.apply(pd.to_numeric, errors='coerce').dropna().sort_values('timestamp')
            
            last_ts = int(df_temp['timestamp'].iloc[-1])
            all_data.extend(df_temp.values.tolist())
            
            since = last_ts + (5 * 60 * 1000)
            if last_ts >= final_end_ts: break
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(2)
            continue

    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = df['timestamp'].astype('int64')
        df = df[(df['timestamp'] >= int(start_dt.timestamp() * 1000)) & (df['timestamp'] <= int(end_dt.timestamp() * 1000))]
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df.to_parquet(file_path, compression='snappy', index=False)
        print(f"[{symbol}] Downloaded {len(df)} candles.")

if __name__ == '__main__':
    for s in TICKERS:
        download_oos(s)
