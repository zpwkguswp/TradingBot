import asyncio
import os
import pandas as pd
from datetime import datetime
import ccxt.async_support as ccxt
import time

# v30_train에서 FULL_UNIVERSE 리스트 임포트
try:
    from v30_train import FULL_UNIVERSE
except ImportError:
    print("[Error] v30_train.py를 찾을 수 없거나 FULL_UNIVERSE가 정의되어 있지 않습니다.")
    # 예비용 리스트 (v30_train.py의 STAGE_3_COINS 기반)
    FULL_UNIVERSE = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
        "BNBUSDT", "DOGEUSDT", "DOTUSDT", "POLUSDT", "LTCUSDT",
        "NEARUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "INJUSDT", "SEIUSDT", "STXUSDT",
        "TIAUSDT", "SUIUSDT", "ORDIUSDT", "WIFUSDT", "PENDLEUSDT",
        "JUPUSDT", "PYTHUSDT", "RENDERUSDT", "FETUSDT", "WLDUSDT",
        "BLURUSDT", "GMXUSDT", "DYDXUSDT", "RUNEUSDT", "ATOMUSDT",
        "ALGOUSDT", "EGLDUSDT", "HBARUSDT", "FLOWUSDT", "RNDRUSDT",
        "AXSUSDT", "SANDUSDT", "MANAUSDT", "CHZUSDT", "APEUSDT",
        "GALAUSDT", "IMXUSDT", "LRCUSDT", "ZILUSDT"
    ]

# 설정
TIMEFRAME = '5m'
START_DATE = "2020-01-01 00:00:00"
START_TS = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
DATA_DIR = "data_storage"
LIMIT = 1000
SLEEP_INTERVAL = 0.3  # 코인별 호출 간격 (병렬 처리 시 더 넉넉하게 설정)
MAX_CONCURRENT_TASKS = 3  # 세마포어: 동시에 수집할 코인 수

# 진행 상황 추적용 전역 변수
processed_count = 0
total_coins = len(FULL_UNIVERSE)

async def scrape_coin(exchange, symbol, semaphore):
    """세마포어를 사용하여 특정 코인의 데이터를 수집하고 저장하는 워커"""
    global processed_count
    
    async with semaphore:
        file_path = os.path.join(DATA_DIR, f"{symbol}_5m_full.parquet")
        existing_df = None
        since = START_TS
        
        # 1. 증분 업데이트 확인: 기존 파일이 있으면 마지막 타임스탬프부터 시작
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_parquet(file_path)
                if not existing_df.empty:
                    since = int(existing_df['timestamp'].max() + 1)
                    # print(f"  [{symbol}] 기존 데이터 발견. {datetime.fromtimestamp(since/1000)} 부터 이어서 수집합니다.")
            except Exception as e:
                print(f"  [{symbol}] 기존 파일 읽기 실패: {e}. 새로 수집합니다.")
        
        new_ohlcv = []
        retry_count = 0
        
        while True:
            try:
                # Bybit에서 데이터 호출
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=LIMIT)
                
                if not ohlcv:
                    break
                
                new_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                
                # 로깅 (터미널 가독성을 위해 한 줄에 표시)
                dt = datetime.fromtimestamp(last_ts / 1000)
                print(f"  > [{symbol}] {dt.strftime('%Y-%m')} 수집 중... (신규: {len(new_ohlcv)} rows)", end='\r')
                
                if last_ts == since: break
                since = last_ts + 1
                
                # 현재 시간에 도달하면 종료
                if last_ts >= (time.time() * 1000) - (5 * 60 * 1000):
                    break
                    
                await asyncio.sleep(SLEEP_INTERVAL)
                retry_count = 0 # 성공 시 리트라이 카운트 초기화
                
            except ccxt.RateLimitExceeded as e:
                print(f"\n  [RateLimit] {symbol}: 20초 대기... ({e})")
                await asyncio.sleep(20)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                print(f"\n  [NetworkError] {symbol}: 10초 대기 후 재시도... ({e})")
                await asyncio.sleep(10)
            except (ccxt.ExchangeError, ccxt.BadSymbol) as e:
                err_msg = str(e).lower()
                if "does not have market symbol" in err_msg or "symbol not found" in err_msg:
                    print(f"\n  [Skip] {symbol}: 거래소에 해당 심볼이 없습니다. (상장폐지 또는 이름변경)")
                    return None # 루프 종료 및 함수 탈출
                print(f"\n  [ExchangeError] {symbol}: 5초 대기 후 재시도... ({e})")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"\n  [Error] {symbol} 수집 중 미확인 에러: {e}. 3초 대기...")
                await asyncio.sleep(3)
                retry_count += 1
                if retry_count > 5:
                    print(f"\n  [Fatal] {symbol} 수집 포기 (5회 연속 에러)")
                    break

        # 2. 데이터 병합 및 저장
        if new_ohlcv:
            new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if existing_df is not None:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df
                
            # 전처리: 중복 제거 및 시간순 정렬
            final_df = final_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            final_df.to_parquet(file_path, index=False)
            print(f"\n  [Saved] {symbol}: 총 {len(final_df)}행 저장 완료.")
        else:
            print(f"\n  [Skip] {symbol}: 추가할 새로운 데이터가 없습니다.")

        processed_count += 1
        print(f"进度: [{processed_count} / {total_coins}] {symbol} 처리 완료")

async def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    print(f"\n{'='*50}")
    print(f"V31 History Scraper Pro 시작 (병렬도: {MAX_CONCURRENT_TASKS})")
    print(f"대상 코인 수: {total_coins}")
    print(f"{'='*50}\n")
    
    try:
        tasks = [scrape_coin(exchange, ticker, semaphore) for ticker in FULL_UNIVERSE]
        await asyncio.gather(*tasks)
    finally:
        await exchange.close()
        print("\n[Done] 모든 수집 프로세스가 종료되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())
