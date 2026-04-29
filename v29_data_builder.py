import os
import pandas as pd
import numpy as np
import pandas_ta as ta

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data_storage"
TIMEFRAMES = ["1h", "2h", "4h"]
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT", "NEARUSDT"]

def _wilder_smooth(s: pd.Series, w: int) -> pd.Series:
    res = np.zeros(len(s))
    if len(s) < w:
        return pd.Series(res, index=s.index)
    res[w - 1] = s.iloc[:w].mean()
    for i in range(w, len(s)):
        res[i] = res[i - 1] * (1 - 1 / w) + s.iloc[i] / w
    return pd.Series(res, index=s.index)

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move   = high - high.shift()
    down_move = low.shift() - low
    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr    = _wilder_smooth(tr, window)
    pos_di = 100 * _wilder_smooth(pos_dm, window) / (atr + 1e-9)
    neg_di = 100 * _wilder_smooth(neg_dm, window) / (atr + 1e-9)
    dx     = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9)
    return _wilder_smooth(dx, window).fillna(0)

def zscore(series: pd.Series, window: int = 96) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)

def add_macro_features(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    print(f"[{tf_name}] Injecting Macro Features (ADX / ATR-Ratio / BB-Width)...")
    df = df.copy()
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["macro_adx"] = adx_df["ADX_14"] / 100.0
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_sma    = atr_series.rolling(window=14).mean()
    df["macro_atr_ratio"] = atr_series / (atr_sma + 1e-9)
    bb_df = ta.bbands(df["close"], length=20, std=2.0)
    bbl_col = [c for c in bb_df.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb_df.columns if c.startswith("BBM")][0]
    bbu_col = [c for c in bb_df.columns if c.startswith("BBU")][0]
    df["macro_bb_width"] = (bb_df[bbu_col] - bb_df[bbl_col]) / (bb_df[bbm_col] + 1e-9)
    return df

def build_indicators(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    print(f"[{tf_name}] Calculating V29 ATR-Normalized Indicators...")
    df = df.copy()
    
    # 🌟 [V29 Core] 절대적 달러(USDT) 변화폭을 제외하기 위한 기초 ATR (14) 계산
    atr_val = ta.atr(df["high"], df["low"], df["close"], length=14).bfill() + 1e-9
    df["atr_abs"] = atr_val

    df["ema_20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema_60"]  = df["close"].ewm(span=60,  adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # 🌟 [V29] 퍼센티지(%) 편차 대신, ATR 배수(Multiples) 편차로 정규화!
    df["disparity_20_atr"]  = (df["close"] - df["ema_20"]) / df["atr_abs"]
    df["disparity_60_atr"]  = (df["close"] - df["ema_60"]) / df["atr_abs"]
    df["disparity_200_atr"] = (df["close"] - df["ema_200"]) / df["atr_abs"]
    
    df["ema_squeeze"]   = df[["ema_20", "ema_60", "ema_200"]].std(axis=1) / df["atr_abs"]
    df["adx_14"]        = calculate_adx(df, 14) / 100.0

    df["hl"] = (df["low"]  > df["low"].shift(1)).astype(float)
    df["hh"] = (df["high"] > df["high"].shift(1)).astype(float)
    df["structural_rev_long"] = ((df["hl"] == 1) & (df["hh"] == 1)).astype(float)
    df["ll"] = (df["low"]  < df["low"].shift(1)).astype(float)
    df["lh"] = (df["high"] < df["high"].shift(1)).astype(float)
    df["structural_rev_short"] = ((df["ll"] == 1) & (df["lh"] == 1)).astype(float)

    # Risk (atr_pct)
    df["atr_raw"] = df["atr_abs"] / df["close"] 
    df["atr_z"]   = zscore(df["atr_raw"], 96)
    
    # 거래량 모멘텀
    price_diff_atr = df["close"].diff() / df["atr_abs"]
    df["volume_log"] = np.log1p(df["volume"])
    df["tfi"]     = zscore((price_diff_atr * df["volume_log"]).rolling(14).mean(), 96)

    rsi_main    = ta.rsi(df["close"], length=3)
    df["crsi"]  = rsi_main.fillna(50) / 100.0
    df["rsi_z"] = zscore(ta.rsi(df["close"], length=14).fillna(50), 96)

    # Placeholders
    df["tcn_p_up"]    = 1.0 / 3.0
    df["tcn_p_down"]  = 1.0 / 3.0
    df["tcn_p_chop"]  = 1.0 / 3.0
    df["tcn_p_var"]   = 0.0
    df["delta_thresh"]= 0.001
    return df

def run_builder(ticker: str):
    input_path = os.path.join(DATA_DIR, f"{ticker}_5m.parquet")
    if not os.path.exists(input_path):
        alt_path = os.path.join(DATA_DIR, f"{ticker}_USDT_5m.parquet")
        if os.path.exists(alt_path): input_path = alt_path
        else:
            print(f"[!] Warning: {input_path} not found. Skipping {ticker}.")
            return

    print(f"[Builder] Loading 5m source: {input_path}")
    df_raw = pd.read_parquet(input_path)

    if "timestamp" in df_raw.columns:
        unit = "ms" if df_raw["timestamp"].iloc[0] > 1e12 else "s"
        df_raw.index = pd.to_datetime(df_raw["timestamp"], unit=unit)
        df_raw.drop(columns=["timestamp"], inplace=True)

    df_raw = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()

    print("[Builder] Pre-calculating 4H for Macro (h1_*) features...")
    agg_logic = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_4h_raw  = df_raw.resample("4h").agg(agg_logic).dropna()
    df_4h      = build_indicators(df_4h_raw, "4h")

    macro_cols   = ["ema_20", "ema_60", "ema_200", "adx_14", "atr_abs"]
    df_4h_macro  = df_4h[macro_cols].rename(columns={c: f"h1_{c}" for c in macro_cols})
    df_4h_macro  = df_4h_macro.shift(1)

    for tf in TIMEFRAMES:
        tf_clean = tf.replace("min", "m")
        df_tf = df_raw.resample(tf).agg(agg_logic).dropna()
        df_tf = build_indicators(df_tf, tf_clean)
        df_tf = add_macro_features(df_tf, tf_clean)
        df_final = df_tf.join(df_4h_macro, how="left").ffill()
        df_final.dropna(inplace=True)

        output_file = f"{ticker}_{tf_clean}.parquet"
        output_path = os.path.join(DATA_DIR, output_file)

        df_final = df_final.reset_index()
        if "index" in df_final.columns and "timestamp" not in df_final.columns:
            df_final.rename(columns={"index": "timestamp"}, inplace=True)

        df_final.to_parquet(output_path, index=False)
        print(f"[Builder] Saved: {output_path}  ({len(df_final)} rows)")

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            print(f"\n" + "="*50)
            print(f"[V29 Builder] >> {ticker} Processing")
            print("="*50)
            run_builder(ticker)
        except Exception as e:
            print(f"[V29 Builder] Error on {ticker}: {e}")
