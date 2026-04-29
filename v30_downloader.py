"""
V30 대함대 병참 기지 (Universal Data Downloader)
═══════════════════════════════════════════════════════════════════
목표: 50개 타겟 코인의 5m 캔들 데이터를 최신 시점까지 동기화하고,
      이후 v29_data_builder.py를 호출하여 1h, 2h, 4h 지표를 일괄 생성합니다.
특징:
 - 기존 파일이 있으면 마지막 시간부터 '이어받기' (초고속 업데이트)
 - 파일이 없으면 최근 120일치 데이터를 '새로 받기'
═══════════════════════════════════════════════════════════════════
"""
import os
import asyncio
import time
import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

# V29 지표 생성기(Builder) 임포트
try:
    from v29_data_builder import run_builder
except ImportError:
    run_builder = None
    print("⚠️ 'v29_data_builder.py'를 찾을 수 없어 다운로드 후 지표 빌드를 건너뜁니다.")

DATA_DIR = "data_storage"
os.makedirs(DATA_DIR, exist_ok=True)

# 🚨 중복 없는 50개 정예 코인 리스트 확정
TARGET_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "BNBUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "NEARUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "SEIUSDT", "STXUSDT",
    "TIAUSDT", "SUIUSDT", "ORDIUSDT", "WIFUSDT", "PENDLEUSDT",
    "JUPUSDT", "PYTHUSDT", "RENDERUSDT", "FETUSDT", "WLDUSDT",
    "BLURUSDT", "GMXUSDT", "DYDXUSDT", "RUNEUSDT", "ATOMUSDT",
    "ALGOUSDT", "EGLDUSDT", "HBARUSDT", "FLOWUSDT", "RNDRUSDT",
    "AXSUSDT", "SANDUSDT", "MANAUSDT", "CHZUSDT", "APEUSDT",
    "GALAUSDT", "IMXUSDT", "LRCUSDT", "ZILUSDT", "KASUSDT"
]

async def fetch_ohlcv_for_coin(exchange, ticker, days=120):
    symbol = f"{ticker[:-4]}/USDT:USDT"
    file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
    
    # 1. 이어받기 vs 새로받기 분기 처리
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            last_ts = int(df['timestamp'].max())
            since = last_ts + 1
            is_new = False
            msg_prefix = "🔄 [업데이트]"
        except Exception as e:
            print(f"⚠️ {ticker} 파일 손상됨. 새로 받습니다: {e}")
            df = pd.DataFrame()
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            is_new = True
            msg_prefix = "📥 [신규다운]"
    else:
        df = pd.DataFrame()
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        is_new = True
        msg_prefix = "📥 [신규다운]"

    new_data = []
    
    while True:
        try:
            # 바이비트 선물의 5m 캔들 요청 (최대 1000개)
            ohlcv = await exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            if not ohlcv or len(ohlcv) <= 1: 
                break
            
            new_data.extend(ohlcv)
            since = int(ohlcv[-1][0]) + 1
            
            # 1000개 미만으로 들어오면 현재 시간까지 다 받은 것임
            if len(ohlcv) < 500: 
                break
                
            # Rate Limit 방어 (바이비트 규정 준수)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f" ❌ [{ticker}] 수집 에러: {e}")
            await asyncio.sleep(2)
            break

    # 2. 데이터 병합 및 저장
    if new_data:
        df_new = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if not df.empty:
            df_final = pd.concat([df, df_new]).drop_duplicates(subset=['timestamp'], keep='last')
        else:
            df_final = df_new
            
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)
        df_final.to_parquet(file_path, index=False)
        print(f"{msg_prefix} {ticker:10s} | 최신 데이터 {len(df_new):5d}건 추가 완료 (총 {len(df_final):6d}건)")
        return True
    else:
        print(f"✅ [최신상태] {ticker:10s} | 이미 가장 최신 데이터입니다.")
        return False

async def main():
    print("=" * 65)
    print("🚀 V30 대함대 병참 기지 가동 (50개 코인 데이터 확보)")
    print("=" * 65)
    
    exchange = ccxt.bybit({'enableRateLimit': True})
    
    try:
        await exchange.load_time_difference()
    except Exception as e:
        print(f"⚠️ 시간 동기화 실패 (무시하고 진행): {e}")

    # 비동기로 하나씩 순차 다운로드 (거래소 밴 방지)
    for i, ticker in enumerate(TARGET_COINS):
        print(f"[{i+1}/{len(TARGET_COINS)}] 데이터 스캔 중: {ticker}...")
        await fetch_ohlcv_for_coin(exchange, ticker, days=120)
    
    await exchange.close()
    print("\n" + "=" * 65)
    print("🎯 모든 5m Raw 데이터 수집/업데이트 완료!")
    print("=" * 65)

    # 3. Builder를 호출하여 1h, 2h, 4h 지표 강제 생성
    if run_builder:
        print("\n⚙️ V29 Builder 엔진 가동: 5m 데이터를 기반으로 거시 지표(1h, 2h, 4h)를 생성합니다.")
        for i, ticker in enumerate(TARGET_COINS):
            print(f"\n▶ [{i+1}/{len(TARGET_COINS)}] {ticker} 지표 조립 중...")
            try:
                run_builder(ticker)
            except Exception as e:
                print(f"❌ {ticker} 빌드 실패: {e}")
                
        print("\n🎖️ 모든 데이터 확보 및 지표 파케이(Parquet) 생성 완료! 훈련을 시작하십시오.")

if __name__ == "__main__":
    asyncio.run(main())