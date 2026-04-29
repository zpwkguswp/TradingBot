"""
==============================================================================
  거래소 모듈 (exchange.py) - V13.1 Master (Time Sync & One-Shot SL/TP)
  - Fix: Bybit 10002 Timestamp Error (강제 시간 동기화 및 recvWindow 확장)
  - Feature: One-Shot SL/TP 동시 전송 지원
==============================================================================
"""

import ccxt.async_support as ccxt
import asyncio
import logging
import pandas as pd
from config import BYBIT_API_KEY, BYBIT_API_SECRET

logger = logging.getLogger(__name__)

class ExchangeClient:
    def __init__(self, api_key=None, secret=None):
        self.exchange = ccxt.bybit({
            'apiKey': api_key or BYBIT_API_KEY,
            'secret': secret or BYBIT_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,  # 💡 CCXT 자체 시차 보정 활성화
                'recvWindow': 60000,              # 💡 바이비트 허용 오차를 60초로 대폭 확장
            }
        })
        self.exchange.options['recvWindow'] = 60000
        self.hedge_mode_set_symbols = set()
        self._time_synced = False  # 강제 동기화 실행 여부 체크

    async def close(self):
        await self.exchange.close()

    # 💡 [핵심 패치] 바이비트 서버와 봇의 시차를 강제로 맞추는 함수
    async def _sync_time_if_needed(self):
        if not self._time_synced:
            try:
                await self.exchange.load_time_difference()
                self._time_synced = True
                logger.info("🕒 [Exchange] 바이비트 서버와 시간 동기화 완료 (10002 에러 방어막 가동)")
            except Exception as e:
                logger.warning(f"⚠️ 시간 동기화 시도 중 에러 발생 (무시하고 진행): {e}")

    async def fetch_balance(self):
        await self._sync_time_if_needed() # 개인정보 요청 전 시계열 확인
        try: return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"⚠️ 잔고 조회 실패: {e}")
            return {}

    async def get_total_equity(self) -> float:
        try:
            bal = await self.fetch_balance()
            return float(bal.get('total', {}).get('USDT', 0))
        except: return 0.0

    async def fetch_ohlcv(self, symbol, timeframe='15m', since=None, limit=100):
        """데이터 수집 실패 방지 - 최대 3회 재시도 (빈 데이터 및 예외 대응)"""
        for attempt in range(1, 4):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv or len(ohlcv) == 0:
                    raise ValueError("Empty OHLCV data received")
                return ohlcv
            except Exception as e:
                wait_time = attempt * 2
                if attempt < 3:
                    logger.warning(f"⚠️ [{symbol}] 데이터 수집 시도 실패 ({attempt}/3) - {wait_time}초 후 재시도... 원인: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"🚨 [{symbol}] 데이터 수집 최종 실패 (3/3) - 원인: {e}")
                    return []
        return []

    async def fetch_ticker(self, symbol):
        return await self.exchange.fetch_ticker(symbol)

    async def get_top_volume_symbols(self, top_n=10, excluded=None):
        try:
            tickers = await self.exchange.fetch_tickers()
            symbols = [s for s, d in tickers.items() if '/USDT:USDT' in s and d.get('quoteVolume') is not None]
            sorted_symbols = sorted(symbols, key=lambda x: tickers[x]['quoteVolume'], reverse=True)
            return sorted_symbols[:top_n]
        except: return ['BTC/USDT:USDT', 'ETH/USDT:USDT']

    async def check_market_mood(self) -> str:
        try:
            positive_score = 0
            btc_ohlcv = await self.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=120)
            if btc_ohlcv:
                df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()
                if df['close'].iloc[-2] > df['ema60'].iloc[-2]: positive_score += 1

            top_symbols = await self.get_top_volume_symbols(top_n=5)
            alt_top_3 = [sym for sym in top_symbols if 'BTC' not in sym][:3]

            for sym in alt_top_3:
                ohlcv = await self.fetch_ohlcv(sym, '1h', limit=120)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()
                    if df['close'].iloc[-2] > df['ema60'].iloc[-2]: positive_score += 1
            
            if positive_score >= 3: return "GREEN"   
            elif positive_score <= 1: return "RED"   
            else: return "NEUTRAL"                  
            
        except Exception as e:
            logger.error(f"⚠️ 신호등 계산 에러: {e}"); return "RED" 

    def format_amount(self, symbol, amount):
        try: return float(self.exchange.amount_to_precision(symbol, amount))
        except: return float(amount)

    def format_price(self, symbol, price):
        try: return float(self.exchange.price_to_precision(symbol, price))
        except: return float(price)

    async def fetch_market_min_amount(self, symbol):
        """거래소의 최소 주문 수량(min qty)을 조회합니다."""
        try:
            await self.exchange.load_markets()
            if symbol in self.exchange.markets:
                market = self.exchange.market(symbol)
                return float(market['limits']['amount']['min'])
            return 0.001 # 안전을 위한 보수적 기본값
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] 최소 수량 조회 실패: {e}")
            return 0.001

    async def _ensure_hedge_mode(self, symbol):
        if symbol in self.hedge_mode_set_symbols: return
        try:
            await self.exchange.set_position_mode(True, symbol)
            self.hedge_mode_set_symbols.add(symbol)
        except: self.hedge_mode_set_symbols.add(symbol)

    async def _ensure_isolated_mode(self, symbol, leverage):
        # 1. Margin Mode 설정 시도 (유저 요청: 격리 모드 강제)
        try: 
            await self.exchange.set_margin_mode('ISOLATED', symbol)
            logger.info(f"🛡️ [{symbol}] 격리(Isolated) 마진 모드 강제 설정 완료")
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" in err_msg or "already" in err_msg or "isolated" in err_msg:
                logger.info(f"🛡️ [{symbol}] 마진 모드 이미 격리(Isolated) 상태 유지 중")
            else:
                logger.warning(f"⚠️ [{symbol}] 격리 모드 변경 에러 (이미 설정되었거나 통합마진 계정일 수 있음): {e}")
            
        # 2. 레버리지 설정 강제 (가장 중요)
        try: 
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"🔧 [{symbol}] 레버리지 {leverage}x 서버 전송 완료")
        except Exception as e:
            err_msg = str(e).lower()
            if "not modified" in err_msg or "not changed" in err_msg:
                # 이미 원하는 레버리지로 설정되어 있는 경우
                logger.info(f"🔧 [{symbol}] 레버리지 이미 {leverage}x로 유지 중")
            else:
                logger.error(f"🚨 [{symbol}] 레버리지 {leverage}x 변경 실패 (바이비트 거부): {e}")
                raise Exception(f"안전 레버리지({leverage}x) 설정 거부됨. 청산 위험으로 진입 포기.")

    async def place_order_with_sl(self, symbol, side, amount, client_id, leverage, sl_price, tp_price=None, position_side='long'):
        await self._sync_time_if_needed() # 주문 전 시계열 확인
        try:
            await self._ensure_hedge_mode(symbol)
            await self._ensure_isolated_mode(symbol, leverage)
            
            pos_idx = 1 if position_side.lower() == 'long' else 2
            
            formatted_sl = self.format_price(symbol, sl_price)
            formatted_tp = self.format_price(symbol, tp_price) if tp_price else None
            formatted_amount = self.format_amount(symbol, amount)

            params = {
                'positionIdx': pos_idx,
                'stopLoss': formatted_sl,
                'tpTriggerBy': 'LastPrice', 
                'slTriggerBy': 'LastPrice', 
                'tpslMode': 'Full',
                'clientOrderId': client_id
            }
            if formatted_tp:
                params['takeProfit'] = formatted_tp

            logger.info(f"⏳ [V13.1] {symbol} 진입 및 방어막(SL/TP) 동시 전송 중...")
            
            order = await self.exchange.create_order(
                symbol=symbol, 
                type='market', 
                side=side, 
                amount=formatted_amount, 
                params=params
            )

            if order:
                logger.info(f"🛡️ [완료] {symbol} {side.upper()} 진입 + SL/TP 방어막 설정 완벽 체결")
            return order

        except Exception as e: 
            logger.error(f"🚨 [주문 실패] {symbol}: {e}")
            raise e

    async def fetch_funding_rate(self, symbol: str) -> float:
        await self._sync_time_if_needed()
        try:
            fr = await self.exchange.fetch_funding_rate(symbol)
            if isinstance(fr, dict):
                return float(fr.get("fundingRate", fr.get("info", {}).get("fundingRate", 0)) or 0.0)
            return 0.0
        except Exception as e:
            logger.warning("⚠️ 펀딩비 조회 실패 %s: %s", symbol, e)
            return 0.0

    async def fetch_open_positions(self):
        await self._sync_time_if_needed() # 포지션 조회 전 시계열 확인
        try:
            positions = await self.exchange.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) > 0]
        except: return []

    async def close_position_market(self, symbol, side, amount, client_id, position_side='long'):
        await self._sync_time_if_needed()
        try:
            close_side = 'sell' if position_side == 'long' else 'buy'
            pos_idx = 1 if position_side.lower() == 'long' else 2
            formatted_amount = self.format_amount(symbol, amount)
            return await self.exchange.create_order(symbol, 'market', close_side, formatted_amount, params={'positionIdx': pos_idx, 'clientOrderId': client_id})
        except Exception as e: 
            logger.error(f"⚠️ 포지션 종료 실패 ({symbol}): {e}")
            raise e # 🚨 [중요] 예외를 다시 던져서 상위에서 상태 삭제를 방지함