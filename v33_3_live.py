import argparse
import asyncio
import json
import logging
import math
import os
import time
import warnings
from datetime import datetime, timezone
import numpy as np
import torch as th
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium import spaces

# V30/V33 공용 컴포넌트 임포트
from v30_train import TCN6LayerExtractor, FULL_UNIVERSE
from config import BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from exchange import ExchangeClient
from telegram_bot import TelegramBot
from v29_env import V29_Universal_Env

warnings.filterwarnings("ignore")

# ==============================================================================
# [V34] Grand Finale Rank 1 Champion Live Engine (Score 111.7)
# ==============================================================================
MAX_POSITIONS = 1        # [Surgical Edit] 단 1종목 집중 저격 (최고 확신도 올인)
ALLOCATION_RATE = 0.95   # [Surgical Edit] 가용 잔고의 95% 투입 (수수료 버퍼 5%)
LEVERAGE = 3             # [Surgical Edit] 3배 레버리지
TIMEFRAME = "5m"         
TF_MINUTES = 5

SL_ATR_COEF = 3.835
TRAIL_ACT = 0.020        # 2% 수익 시 트레일링 스탑 활성화
PRESERVATION_RATIO = 0.7 # 수익의 70% 보존 (고점 대비 30% 하락 시 익절)
TARGET_PROFIT = 0.008

# [Surgical Edit] V34 Grand Finale Rank 1 Champion (Score 111.7) 적용
MODEL_PATH = "elite_weights/v34_snapshots/v34_elite_rank_score111.7_step29900000"
STATE_FILE = "v33_3_live_state.json"
DATA_DIR = "data_storage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def ticker_to_symbol(ticker: str) -> str:
    return f"{ticker[:-4]}/USDT:USDT" if ticker.endswith("USDT") else f"{ticker}/USDT:USDT"

def next_5m_close_unix() -> float:
    now = time.time()
    period = float(TF_MINUTES * 60)
    return math.ceil(now / period) * period

class V33_LiveEnv(V29_Universal_Env):
    def set_live_data(self, df):
        df = df.copy()
        df["returns"]    = df["close"].pct_change().fillna(0)
        df["log_volume"] = np.log1p(df["volume"])
        df["hl_ratio"]   = (df["high"] - df["low"]) / (df["close"] + 1e-9)

        # [IMMUTABLE CORE] v34_train._load_coin_data와 100% 동일한 피처 엔지니어링
        # v34 Rank 1 모델의 학습 환경을 그대로 재현합니다. 절대 변경 금지.

        # 1. 1시간봉 EMA (v34_train 동일: s in [20, 60, 200])
        for s in [20, 60, 200]:
            df[f"h1_ema_{s}"] = df["close"].ewm(span=s * 12, adjust=False).mean()

        # 2. h1_ema_200 덮어쓰기 (v34_train 동일)
        df["actual_h1_ema_200"] = df["h1_ema_200"].copy()
        df["h1_ema_200"] = df["close"]

        # 3. ATR - [v34_train 동일] 단순 H-L 방식 (True Range 아님)
        # v34_train 주석: "Live와 100% 동기화: True Range -> 단순 H-L 방식으로 교체"
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)

        # 4. 나머지 지표는 v34_train에 없으므로 V29_Universal_Env 기본값을 그대로 사용
        # (adx_14, tcn_p_up 등은 컬럼 없을 시 row.get()의 기본값으로 자동 처리됨)

        self.df = df


class V33LiveBot:
    def __init__(self, is_dry_run: bool = False) -> None:
        self.is_dry_run = is_dry_run
        self.exchange = ExchangeClient(BYBIT_API_KEY, BYBIT_API_SECRET)
        self.telegram = TelegramBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, self.exchange, None)
        self.positions = {}
        self.is_running = True
        
        custom_objects = {"features_extractor_class": TCN6LayerExtractor}
        self.model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
        # [Surgical Edit] 신규 Rank 1 모델 지표 업데이트
        logger.info(f"✅ V34 Rank 1 (Score 111.7, WR 69.7%, PF 7.09) 로드 완료")
        self._load_state()

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f: self.positions = json.load(f).get("positions", {})
            except: self.positions = {}

    def _save_state(self):
        if self.is_dry_run: return
        with open(STATE_FILE, "w") as f: json.dump({"positions": self.positions, "ts": time.time()}, f, indent=2)

    async def fetch_missing_5m_data(self, ticker: str):
        """Parquet 데이터 실시간 동기화 (Empty Data 대응 패치)"""
        symbol = ticker_to_symbol(ticker)
        file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
        try:
            if symbol not in self.exchange.exchange.markets: return False
            
            # 기존 데이터 로드
            df_old = pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
            last_ts = int(df_old['timestamp'].max()) if not df_old.empty else 0

            # 누락분 fetch
            since = max(0, last_ts + 1)
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            except Exception as e:
                # 데이터 수집 실패 시, 기존 데이터가 충분하면 그냥 진행 (유연한 대처)
                if not df_old.empty and len(df_old) > 100:
                    return True 
                return False

            if ohlcv:
                df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # [IMMUTABLE CORE] Sim-to-Real Fix 1: EMA 2400 수렴을 위해 무조건 10,000개 유지!
                # 훈련 환경과 관측치를 일치시키기 위한 최저 방어선입니다. 절대 2,000으로 줄이지 마십시오.
                df_final = df_final.iloc[-10000:] 
                
                df_final.to_parquet(file_path, index=False)
                return True
            
            # 데이터가 Empty여도 기존 데이터가 100개 이상이면 분석 진행 허용
            return not df_old.empty and len(df_old) > 100
        except Exception as e:
            logger.warning(f"⚠️ [{ticker}] 데이터 동기화 건너뜀: {e}")
            return False

    async def sync_with_exchange(self):
        """거래소 실시간 상황으로 장부(Memory)를 완전히 재건합니다."""
        try:
            real_pos = await self.exchange.fetch_open_positions()
            new_positions = {}
            found_tickers = []

            for p in real_pos:
                # 티커 추출 (규격 통일: "ETH/USDT:USDT" -> "ETHUSDT")
                symbol = p.get('symbol', '')
                ticker = symbol.split(':')[0].replace('/', '')
                found_tickers.append(ticker)

                # 기존 데이터가 있으면 보존, 없으면 신규 입양
                if ticker in self.positions:
                    new_positions[ticker] = self.positions[ticker]
                    # 수량과 진입가는 최신 데이터로 업데이트
                    new_positions[ticker].update({
                        "amount": abs(float(p.get('contracts', 0))),
                        "entry_price": float(p.get('entryPrice', 0))
                    })
                else:
                    side = p.get('side', 'long')
                    new_positions[ticker] = {
                        "side": side,
                        "amount": abs(float(p.get('contracts', 0))),
                        "entry_price": float(p.get('entryPrice', 0)),
                        "mfe": 0.0
                    }

            # 장부 교체 및 저장
            self.positions = new_positions
            self._save_state()
            
            if found_tickers:
                logger.info(f"🔍 [Sync] 감지된 포지션({len(found_tickers)}개): {', '.join(found_tickers)}")
            elif len(self.positions) > 0:
                logger.warning(f"⚠️ [Sync] 이전 포지션 존재하나 현재 거래소에서 감지된 포지션 없음.")
        except Exception as e:
            logger.error(f"⚠️ 동기화 실패: {e}")

    async def get_v33_observation(self, ticker: str):
        coin_id = FULL_UNIVERSE.index(ticker) if ticker in FULL_UNIVERSE else 0
        file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
        df_raw = pd.read_parquet(file_path)
        env = V33_LiveEnv(DATA_DIR, [f"{ticker}_5m.parquet"], coin_id)
        env.set_live_data(df_raw)
        obs_history = []
        env.current_step = max(0, len(env.df) - 100)
        while env.current_step < len(env.df):
            obs = env._get_obs()
            if len(obs) == 33: obs = np.append(obs, [0, 0, 0, float(coin_id)])
            obs_history.append(obs)
            env.current_step += 1
        stacked_obs = np.concatenate(obs_history[-4:]).reshape(1, -1)
        return stacked_obs

    async def run(self):
        print("-------------------------------------------------------")
        print("  V34 Grand Finale Live Engine Starting")
        # [Surgical Edit] HUD 정보 업데이트
        print("  Model: V34 Rank 1 Champion (Score 111.7)")
        print("-------------------------------------------------------")
        
        logger.info("📡 순정 텔레그램 통신망 연결 준비 완료.")
        logger.info("🚀 V34 Rank 1 Champion Live Engine Ignition")
        
        # [Harness Protocol] 부팅 즉시 거래소 상황을 강제 입양하여 메모리 일치화
        await self.sync_with_exchange()
        
        while self.is_running:
            try:
                next_close = next_5m_close_unix()
                
                # [IMMUTABLE CORE] Sim-to-Real Fix 2: 캔들 마감 전 조기 출발 완벽 차단! (next_close - 10 아님)
                # 정각 마감을 끝까지 기다려 훈련장과 동일한 완성된 캔들을 참조합니다.
                while time.time() < next_close:
                    await self.sync_with_exchange()
                    
                    if self.positions:
                        # 한 번의 API 호출로 모든 코인 현재가 조회
                        tickers = await self.exchange.exchange.fetch_tickers([ticker_to_symbol(t) for t in self.positions.keys()])
                        for ticker, pos in list(self.positions.items()):
                            symbol = ticker_to_symbol(ticker)
                            tick = tickers.get(symbol)
                            if not tick: continue
                            curr_px = float(tick['last'])
                            
                            # 수익률 및 MFE 계산
                            pnl = (curr_px / pos["entry_price"] - 1.0) if pos["side"] == "long" else (1.0 - curr_px / pos["entry_price"])
                            if pnl > pos.get("mfe", 0.0): pos["mfe"] = pnl
                            
                            # 트레일링 스탑 체크
                            if pos.get("mfe", 0.0) >= TRAIL_ACT:
                                if pnl < pos["mfe"] * PRESERVATION_RATIO:
                                    logger.info(f"💰 [{ticker}] Trailing Stop 발동 (PnL: {pnl*100:.2f}%)")
                                    await self.exchange.close_position_market(symbol, None, pos["amount"], f"TS_{ticker}", position_side=pos["side"])
                                    del self.positions[ticker]; self._save_state()
                                    await self.telegram.send_message(f"💰 **[{ticker}]** 트레일링 스탑 익절! ({pnl*100:.2f}%)")
                    
                    # [IMMUTABLE CORE] 마감 전 촘촘한 감시를 위해 5초 대기
                    await asyncio.sleep(5)
                
                # [IMMUTABLE CORE] Sim-to-Real Fix 2: 정각 직후 거래소 API 갱신 지연 대비 3초 대기 (절대 삭제 금지)
                await asyncio.sleep(3)
                
                # [Harness] 사냥 시작 전 최종 동기화
                await self.sync_with_exchange()
                logger.info(f"🌙 [5m Boundary] 데이터 동기화 및 사냥 시작 (현재: {len(self.positions)}/{MAX_POSITIONS})")
                await self.exchange.exchange.load_markets()

                # ── Phase 1: 전 코인 스캔 - 1.0 신호 후보 및 원시 확신도 수집 ──
                candidates = []  # (ticker, raw_score)
                for ticker in FULL_UNIVERSE[:50]:
                    await asyncio.sleep(0.1) # API 과속 방지
                    if not await self.fetch_missing_5m_data(ticker): continue
                    try:
                        obs = await self.get_v33_observation(ticker)
                        action, _ = self.model.predict(obs, deterministic=True)
                        act_val = float(action[0])

                        if ticker in self.positions:
                            # [Surgical Edit] 이산형 모델에 맞춘 긴급 탈출 로직
                            # 롱(Long)을 들고 있는데 AI가 2.0(Short/하락)을 외치면 즉각 도망칩니다!
                            pos = self.positions[ticker]
                            if pos["side"] == "long" and act_val == 2.0:
                                await self.exchange.close_position_market(ticker_to_symbol(ticker),
                                    None, pos["amount"], f"Flip_{ticker}_{int(time.time())}", position_side=pos["side"])
                                del self.positions[ticker]; self._save_state()
                                await self.telegram.send_message(f"🔴 **[{ticker}]** 하락 위험(2.0) 감지! 롱 포지션 긴급 청산")

                        elif len(self.positions) < MAX_POSITIONS and act_val == 1.0:
                            # [IMMUTABLE CORE] v34_train과 동일: act_val == 1.0 = 롱 신호

                            # [Surgical Edit] Tensor Device 할당 및 이산형 모델의 확률(Probs) 추출
                            # obs_tensor를 모델이 위치한 디바이스(CPU/GPU)로 정확히 올려줍니다.
                            obs_tensor = th.tensor(obs, dtype=th.float32).to(self.model.device)

                            with th.no_grad():
                                dist = self.model.policy.get_distribution(obs_tensor)
                                # Categorical 분포에서 각 액션(0,1,2) 확률표(.probs)를 가져옵니다.
                                # shape=(1,3), 인덱스 1 = 롱(Long) 확신도
                                probs = dist.distribution.probs
                                raw_score = float(probs[0, 1].item())

                            candidates.append((ticker, raw_score))
                            logger.info(f"🎯 [{ticker}] 후보 등록 (Long 확신도: {raw_score*100:.2f}%)") 

                    except Exception as e: logger.error(f"[{ticker}] 오류: {e}")

                # ── Phase 2: 최고 확신도 단 1종목에 올인 ──
                if candidates and len(self.positions) < MAX_POSITIONS:
                    # [리포트] 상위 5개 후보를 Long 확신도(%) 순으로 정렬하여 보고
                    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    top5 = sorted_candidates[:5]
                    
                    top5_log = "\n".join([f"  {i+1}. {t} | Long 확신도: {s*100:.2f}%" for i, (t, s) in enumerate(top5)])
                    logger.info(f"📊 [Top-5 Long 확신도 랭킹]\n{top5_log}")
                    
                    top5_msg = "📊 **[Top-5 Long 확신도 랭킹]**\n" + "\n".join(
                        [f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}위'} `{t}` → `{s*100:.2f}%`"
                         for i, (t, s) in enumerate(top5)]
                    )
                    await self.telegram.send_message(top5_msg)

                    best_ticker, best_score = sorted_candidates[0]
                    logger.info(f"🏆 최고 확신도 타겟: {best_ticker} ({best_score*100:.2f}%) / 후보 {len(candidates)}개 중 선발")
                    await self.telegram.send_message(f"🔍 **[스캔 완료]** {len(candidates)}개 후보 중 **{best_ticker}** 선발 (Long 확신도: {best_score*100:.2f}%)")

                    try:
                        equity    = await self.exchange.get_total_equity()
                        available = await self.exchange.get_available_balance()
                        margin    = min(equity * ALLOCATION_RATE, available * 0.95)
                        if margin < 1.0:
                            logger.warning(f"💰 [Balance Guard] 가용 잔고($ {available:.2f}) 부족으로 사냥 중단")
                        else:
                            symbol = ticker_to_symbol(best_ticker)
                            tick   = await self.exchange.fetch_ticker(symbol)
                            curr_px = float(tick['last'])
                            current_leverage = LEVERAGE

                            raw_amount = (margin * current_leverage) / curr_px
                            amount     = self.exchange.format_amount(symbol, raw_amount)
                            min_amount = await self.exchange.fetch_market_min_amount(symbol)

                            if amount >= min_amount:
                                ohlcv  = await self.exchange.fetch_ohlcv(symbol, "5m", limit=30)
                                df_atr = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                                atr    = (df_atr['h'] - df_atr['l']).rolling(14).mean().iloc[-1]
                                sl_px  = curr_px - SL_ATR_COEF * atr  # 롱 전용

                                if not self.is_dry_run:
                                    order = await self.exchange.place_order_with_sl(
                                        symbol, "buy", amount,
                                        f"V34_{best_ticker}_{int(time.time())}",
                                        current_leverage, sl_px, position_side="long"
                                    )
                                    if order:
                                        self.positions[best_ticker] = {"side": "long", "amount": amount, "entry_price": curr_px, "entry_timestamp": time.time(), "mfe": 0.0}
                                        self._save_state()
                                        msg = (f"🟢 **[V34 최고확신도 진입]** {best_ticker} @ {curr_px:.4f}\n"
                                               f"Long 확신도: {best_score*100:.2f}% | 레버리지: {current_leverage}x | 증거금: ${margin:.2f}")
                                        await self.telegram.send_message(msg)
                                        logger.info(f"🚀 [{best_ticker}] 사냥 개시! (Long 확신도: {best_score*100:.2f}%)")
                    except Exception as e:
                        if "110007" in str(e) or "not enough" in str(e).lower():
                            logger.warning(f"💰 [Balance Guard] 진입 실패: 잔고 부족.")
                        else:
                            logger.error(f"🚨 [{best_ticker}] 주문 실패: {e}")

                await asyncio.sleep(10)
            except Exception as e: logger.error(f"🛑 오류: {e}"); await asyncio.sleep(30)

async def main():
    bot = V33LiveBot(); await bot.run()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("🛑 Live Stop.")
