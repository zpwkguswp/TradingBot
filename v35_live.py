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

# V30/V35 공용 컴포넌트 임포트
from v30_train import TCN6LayerExtractor, FULL_UNIVERSE
from config import BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from exchange import ExchangeClient
from telegram_bot import TelegramBot
from v29_env import V29_Universal_Env

warnings.filterwarnings("ignore")

# ==============================================================================
# [V35] Long/Short Sniper Live Engine (Score 240.7)
# ==============================================================================
MAX_POSITIONS = 1        # 단 1종목 집중 저격 (최고 확신도 올인)
ALLOCATION_RATE = 0.95   # 가용 잔고의 95% 투입 (수수료 버퍼 5%)
LEVERAGE = 10             # 하드코딩 레버리지 (동적 레버리지 미사용 시 기본값)
# TARGET_RISK는 더 이상 고정값으로 사용하지 않습니다 (BTC 실시간 변동성에 연동됨)
TIMEFRAME = "5m"
TF_MINUTES = 5

SL_ATR_COEF = 5 #3.835
TRAIL_ACT = 0.010        # 2% 수익 시 트레일링 스탑 활성화, btc10배 기준시 0.01, 3배 기준지 0.02
PRESERVATION_RATIO = 0.7 # 수익의 70% 보존 (고점 대비 30% 하락 시 익절)
TARGET_PROFIT = 0.008
FLIP_PROB_THRESHOLD = 0.70  # [Surgical Edit] 반전 청산 최소 확신도 방어선 (70% 미만 신호는 무시)
MIN_ENTRANCE_PROB = 0.8    # [Surgical Edit] 진입 문턱: 확신도 98% 이상일 때만 사격 개시

# [V35] Rank 1 Champion (Score 240.7, Step 14,600,000) 적용
MODEL_PATH = "elite_weights/v35_snapshots/v35_elite_rank_score240.7_step14600000"
STATE_FILE = "v35_live_state.json"
DATA_DIR = "data_storage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def ticker_to_symbol(ticker: str) -> str:
    return f"{ticker[:-4]}/USDT:USDT" if ticker.endswith("USDT") else f"{ticker}/USDT:USDT"

def next_5m_close_unix() -> float:
    now = time.time()
    period = float(TF_MINUTES * 60)
    return math.ceil(now / period) * period

# ==============================================================================
# [V35] 실전 동기화 환경 - v35_train._load_coin_data와 100% 동일
# ==============================================================================
class V35_LiveEnv(V29_Universal_Env):
    def set_live_data(self, df):
        df = df.copy()
        # [IMMUTABLE CORE] v35_train._load_coin_data와 100% 동일한 피처 엔지니어링
        # 절대 변경 금지.

        # 0. 시계열 정렬 (v35_train: sort_values("timestamp") 동일)
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["returns"]    = df["close"].pct_change().fillna(0)
        df["log_volume"] = np.log1p(df["volume"])
        df["hl_ratio"]   = (df["high"] - df["low"]) / (df["close"] + 1e-9)

        # 1. 1시간봉 EMA (v35_train 동일: s in [20, 60, 200], span=s*12)
        for s in [20, 60, 200]:
            df[f"h1_ema_{s}"] = df["close"].ewm(span=s * 12, adjust=False).mean()

        # 2. h1_ema_200 덮어쓰기 (v35_train 동일)
        df["actual_h1_ema_200"] = df["h1_ema_200"].copy()
        df["h1_ema_200"] = df["close"]

        # 3. ATR - [v35_train 동일] 단순 H-L 방식 (True Range 아님)
        df["atr_raw"] = (df["high"] - df["low"]).rolling(14).mean() / (df["close"] + 1e-9)

        # 4. NaN 제거 (v35_train: df.dropna().reset_index(drop=True) 동일)
        df = df.dropna().reset_index(drop=True)

        self.df = df


class V35LiveBot:
    def __init__(self, is_dry_run: bool = False) -> None:
        self.is_dry_run = is_dry_run
        self.exchange = ExchangeClient(BYBIT_API_KEY, BYBIT_API_SECRET)
        self.telegram = TelegramBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, self.exchange, None)
        self.positions = {}
        self.is_running = True

        custom_objects = {"features_extractor_class": TCN6LayerExtractor}
        self.model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
        logger.info(f"✅ V35 Rank 1 (Score 240.7, Step 14.6M) 로드 완료 — Long/Short Sniper 준비")
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
        """Parquet 데이터 실시간 동기화"""
        symbol = ticker_to_symbol(ticker)
        file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
        try:
            if symbol not in self.exchange.exchange.markets: return False

            df_old = pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
            last_ts = int(df_old['timestamp'].max()) if not df_old.empty else 0

            since = max(0, last_ts + 1)
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            except Exception as e:
                if not df_old.empty and len(df_old) > 100:
                    return True
                return False

            if ohlcv:
                df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

                # [IMMUTABLE CORE] EMA 2400 수렴을 위해 10,000개 유지 (절대 삭제 금지)
                df_final = df_final.iloc[-10000:]

                df_final.to_parquet(file_path, index=False)
                return True

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
                symbol = p.get('symbol', '')
                ticker = symbol.split(':')[0].replace('/', '')
                found_tickers.append(ticker)

                if ticker in self.positions:
                    new_positions[ticker] = self.positions[ticker]
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

            self.positions = new_positions
            self._save_state()

            if found_tickers:
                logger.info(f"🔍 [Sync] 감지된 포지션({len(found_tickers)}개): {', '.join(found_tickers)}")
            elif len(self.positions) > 0:
                logger.warning(f"⚠️ [Sync] 이전 포지션 존재하나 현재 거래소에서 감지된 포지션 없음.")
        except Exception as e:
            logger.error(f"⚠️ 동기화 실패: {e}")

    async def get_v35_observation(self, ticker: str):
        """v35_train과 동일한 37차원 관측값을 생성합니다."""
        coin_id = FULL_UNIVERSE.index(ticker) if ticker in FULL_UNIVERSE else 0
        file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
        df_raw = pd.read_parquet(file_path)
        env = V35_LiveEnv(DATA_DIR, [f"{ticker}_5m.parquet"], coin_id)
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

    async def _close_position(self, ticker: str, pos: dict, reason: str):
        """포지션을 시장가로 청산하고 상태를 업데이트합니다."""
        symbol = ticker_to_symbol(ticker)
        order_id = f"V35_{reason}_{ticker}_{int(time.time()*1000)}"
        await self.exchange.close_position_market(symbol, None, pos["amount"], order_id, position_side=pos["side"])
        del self.positions[ticker]
        self._save_state()

    async def monitor_resources(self):
        """서버 자원(Disk, RAM)을 감시하고 위험 시 텔레그램 알림을 보냅니다."""
        try:
            import psutil
            # 디스크 체크 (루트 파티션)
            disk = psutil.disk_usage('/')
            # 메모리 체크 (Swap 포함 실질 가용량)
            mem = psutil.virtual_memory()
            
            status_msg = (f"📊 **[서버 자원 리포트]**\n"
                          f"• 디스크: {disk.percent}% 사용 중 ({disk.free/1024**3:.1f}GB 남음)\n"
                          f"• 메 모 리: {mem.percent}% 사용 중")
            
            # 위험 수위 체크 (90% 초과 시)
            if disk.percent > 90 or mem.percent > 95:
                warning_msg = f"⚠️ **[위험] 서버 자원 부족 경고!**\n{status_msg}\n관리자의 확인이 필요합니다."
                logger.warning(warning_msg)
                await self.telegram.send_message(warning_msg)
            else:
                logger.info(f"Healthy: Disk {disk.percent}%, RAM {mem.percent}%")
                
            return status_msg
        except Exception as e:
            logger.error(f"자원 모니터링 중 오류: {e}")
            return None

    async def run(self):
        print("-------------------------------------------------------")
        print("  V35 Long/Short Sniper Live Engine Starting")
        print(f"  Model: V35 Rank 1 Champion (Score 240.7, Step 14.6M)")
        print("-------------------------------------------------------")

        logger.info("📡 순정 텔레그램 통신망 연결 준비 완료.")
        logger.info("🚀 V35 Long/Short Sniper Live Engine Ignition")

        # 부팅 시 첫 리소스 리포트 발송
        sys_status = await self.monitor_resources()
        if sys_status:
            await self.telegram.send_message(f"🚀 **V35 엔진 가동 시작**\n{sys_status}")

        # [Harness Protocol] 부팅 즉시 거래소 상황을 강제 입양하여 메모리 일치화
        await self.sync_with_exchange()

        last_resource_check = 0
        while self.is_running:
            try:
                # 1시간마다 자원 체크
                if time.time() - last_resource_check > 3600:
                    await self.monitor_resources()
                    last_resource_check = time.time()
                next_close = next_5m_close_unix()

                # [IMMUTABLE CORE] 정각 마감을 끝까지 기다려 훈련장과 동일한 완성된 캔들을 참조합니다.
                while time.time() < next_close:
                    await self.sync_with_exchange()

                    if self.positions:
                        tickers = await self.exchange.exchange.fetch_tickers([ticker_to_symbol(t) for t in self.positions.keys()])
                        for ticker, pos in list(self.positions.items()):
                            symbol = ticker_to_symbol(ticker)
                            tick = tickers.get(symbol)
                            if not tick: continue
                            curr_px = float(tick['last'])

                            # 롱/숏 방향에 따른 수익률 계산
                            if pos["side"] == "long":
                                pnl = (curr_px / pos["entry_price"] - 1.0)
                            else:
                                pnl = (1.0 - curr_px / pos["entry_price"])

                            if pnl > pos.get("mfe", 0.0): pos["mfe"] = pnl

                            # 트레일링 스탑 체크
                            if pos.get("mfe", 0.0) >= TRAIL_ACT:
                                if pnl < pos["mfe"] * PRESERVATION_RATIO:
                                    logger.info(f"💰 [{ticker}] Trailing Stop 발동 (Side: {pos['side']}, PnL: {pnl*100:.2f}%)")
                                    try:
                                        await self._close_position(ticker, pos, "TS")
                                        await self.telegram.send_message(f"💰 **[{ticker}]** 트레일링 스탑 익절! ({pos['side']}, {pnl*100:.2f}%)")
                                    except Exception as e:
                                        logger.warning(f"⚠️ 포지션 종료 실패 ({ticker_to_symbol(ticker)}): {e}")

                    # [IMMUTABLE CORE] 마감 전 촘촘한 감시를 위해 5초 대기
                    await asyncio.sleep(5)

                # [IMMUTABLE CORE] 정각 직후 거래소 API 갱신 지연 대비 3초 대기 (절대 삭제 금지)
                await asyncio.sleep(3)

                # [Harness] 사냥 시작 전 최종 동기화
                await self.sync_with_exchange()
                logger.info(f"🌙 [5m Boundary] 데이터 동기화 및 사냥 시작 (현재: {len(self.positions)}/{MAX_POSITIONS})")
                await self.exchange.exchange.load_markets()

                # ── Phase 1: 전 코인 스캔 ──
                # (ticker, raw_score, score_label, action_type)
                candidates = []
                for ticker in FULL_UNIVERSE[:10]:
                    await asyncio.sleep(2.5)  # API 과속 방지 (Bybit Rate Limit 10006 에러 완벽 차단용 2.5s)
                    if not await self.fetch_missing_5m_data(ticker): continue
                    try:
                        obs = await self.get_v35_observation(ticker)

                        # [Harness] 모든 코인의 확률을 선제적으로 추출하여 매 5분마다 무조건 로깅
                        obs_tensor = th.tensor(obs, dtype=th.float32).to(self.model.device)
                        with th.no_grad():
                            dist = self.model.policy.get_distribution(obs_tensor)
                            probs = dist.distribution.probs
                            prob_hold  = float(probs[0, 0].item())
                            prob_long  = float(probs[0, 1].item())
                            prob_short = float(probs[0, 2].item())
                            
                        logger.info(f"🔎 [{ticker}] 확률분포 - 관망:{prob_hold*100:.1f}% | 롱:{prob_long*100:.1f}% | 숏:{prob_short*100:.1f}%")

                        # [Surgical Edit] 예측 함수 블랙박스를 피하고, 추출된 확률에서 직접 최고값을 도출
                        act_val = int(th.argmax(probs, dim=1).item())

                        # ── 포지션 반전 청산 로직 (확률 기반 정교화) ──
                        if ticker in self.positions:
                            pos = self.positions[ticker]

                            # 롱 보유 중, 숏 확률이 방어선(70%) 초과 시에만 긴급 탈출
                            if pos["side"] == "long" and prob_short > FLIP_PROB_THRESHOLD:
                                logger.info(f"🔄 [{ticker}] 롱→숏 반전 위험({prob_short*100:.1f}%)! 긴급 청산")
                                try:
                                    await self._close_position(ticker, pos, "Flip")
                                    await self.telegram.send_message(f"🔴 **[{ticker}]** 숏 위험({prob_short*100:.1f}%) 감지! 긴급 탈출")
                                except Exception as e:
                                    logger.warning(f"⚠️ 반전 청산 실패 ({ticker}): {e}")

                            # 숏 보유 중, 롱 확률이 방어선(70%) 초과 시에만 긴급 탈출
                            elif pos["side"] == "short" and prob_long > FLIP_PROB_THRESHOLD:
                                logger.info(f"🔄 [{ticker}] 숏→롱 반전 위험({prob_long*100:.1f}%)! 긴급 청산")
                                try:
                                    await self._close_position(ticker, pos, "Flip")
                                    await self.telegram.send_message(f"🟢 **[{ticker}]** 롱 위험({prob_long*100:.1f}%) 감지! 긴급 탈출")
                                except Exception as e:
                                    logger.warning(f"⚠️ 반전 청산 실패 ({ticker}): {e}")

                        # ── 신규 진입 후보 수집 (슬롯 여유 시) ──
                        elif len(self.positions) < MAX_POSITIONS and act_val in (1, 2):
                            if act_val == 1:
                                raw_score = prob_long
                                score_label = f"Long 확신도: {raw_score*100:.2f}%"
                                action_type = "long"
                            else:
                                raw_score = prob_short
                                score_label = f"Short 확신도: {raw_score*100:.2f}%"
                                action_type = "short"

                            candidates.append((ticker, raw_score, score_label, action_type))
                            logger.info(f"🎯 [{ticker}] 후보 등록 ({score_label})")

                    except Exception as e:
                        logger.error(f"[{ticker}] 오류: {e}")

                # ── Phase 2: 최고 확신도 단 1종목에 올인 ──
                if candidates and len(self.positions) < MAX_POSITIONS:
                    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    top5 = sorted_candidates[:5]

                    top5_log = "\n".join([f"  {i+1}. {t} | {label}" for i, (t, s, label, a) in enumerate(top5)])
                    logger.info(f"📊 [Top-5 확신도 랭킹]\n{top5_log}")

                    top5_msg = "📊 **[Top-5 확신도 랭킹]**\n" + "\n".join(
                        [f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}위'} `{t}` → `{label}`"
                         for i, (t, s, label, a) in enumerate(top5)]
                    )
                    await self.telegram.send_message(top5_msg)

                    best_ticker, best_score, best_label, best_action = sorted_candidates[0]
                    
                    # ── [Surgical Edit] 진입 문턱(Confidence Guard) 검사 ──
                    if best_score < MIN_ENTRANCE_PROB:
                        msg = f"⚠️ **[관망]** 최고 후보 {best_ticker}({best_score*100:.1f}%)가 진입 문턱({MIN_ENTRANCE_PROB*100:.1f}%)에 미달하여 사격을 중지합니다."
                        logger.warning(msg)
                        await self.telegram.send_message(msg)
                        continue

                    logger.info(f"🏆 최고 확신도 타겟: {best_ticker} ({best_label}) / 후보 {len(candidates)}개 중 선발")
                    await self.telegram.send_message(f"🔍 **[스캔 완료]** {len(candidates)}개 후보 중 **{best_ticker}** 선발 ({best_label})")

                    try:
                        equity    = await self.exchange.get_total_equity()
                        available = await self.exchange.get_available_balance()
                        margin    = min(equity * ALLOCATION_RATE, available * 0.95)
                        if margin < 1.0:
                            logger.warning(f"💰 [Balance Guard] 가용 잔고($ {available:.2f}) 부족으로 사냥 중단")
                        else:
                            symbol  = ticker_to_symbol(best_ticker)
                            tick    = await self.exchange.fetch_ticker(symbol)
                            curr_px = float(tick['last'])

                            # ── [1] 타겟 코인 ATR 계산 ──
                            ohlcv  = await self.exchange.fetch_ohlcv(symbol, "5m", limit=30)
                            df_atr = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                            atr    = (df_atr['h'] - df_atr['l']).rolling(14).mean().iloc[-1]
                            atr_pct = atr / (curr_px + 1e-9)  # 현재가 대비 ATR 비율

                            # ── [2] 동적 레버리지 계산 (BTC 10x Risk Parity) ──
                            # [IMMUTABLE CORE] 사령관님 지시: BTC는 무조건 10배로 고정. 절대 수정 불가.
                            # 타 코인들의 레버리지는 BTC 10배 기준 리스크(변동성)에 수학적으로 동기화시킴.
                            btc_ohlcv = await self.exchange.fetch_ohlcv("BTC/USDT:USDT", "5m", limit=30)
                            df_btc    = pd.DataFrame(btc_ohlcv, columns=['t','o','h','l','c','v'])
                            btc_atr   = (df_btc['h'] - df_btc['l']).rolling(14).mean().iloc[-1]
                            btc_px    = float(df_btc['c'].iloc[-1])
                            btc_atr_pct = btc_atr / (btc_px + 1e-9)

                            # BTC의 변동성 * 10배 = 시스템 전체의 목표 변동성(Target Risk)
                            dynamic_target_risk = btc_atr_pct * 10.0

                            if atr_pct > 0:
                                raw_lev = dynamic_target_risk / atr_pct
                                current_leverage = int(max(1, min(10, round(raw_lev))))
                            else:
                                current_leverage = LEVERAGE  # fallback

                            logger.info(f"📈 [Dynamic Lev] BTC 기준 ATR: {btc_atr_pct*100:.3f}% (10x 적용) | 타겟({best_ticker}) ATR: {atr_pct*100:.3f}% → 동적 레버리지: {current_leverage}x")

                            # ── [3] 수량 및 손절가 산출 ──
                            raw_amount = (margin * current_leverage) / curr_px
                            amount     = self.exchange.format_amount(symbol, raw_amount)
                            min_amount = await self.exchange.fetch_market_min_amount(symbol)

                            if amount >= min_amount:
                                # 롱/숏에 따라 손절가 방향 반전
                                if best_action == "long":
                                    sl_px     = curr_px - SL_ATR_COEF * atr
                                    order_dir = "buy"
                                    emoji     = "🟢"
                                else:
                                    sl_px     = curr_px + SL_ATR_COEF * atr
                                    order_dir = "sell"
                                    emoji     = "🔴"

                                if not self.is_dry_run:
                                    order_id = f"V35_{best_action.upper()}_{best_ticker}_{int(time.time()*1000)}"
                                    order = await self.exchange.place_order_with_sl(
                                        symbol, order_dir, amount,
                                        order_id,
                                        current_leverage, sl_px, position_side=best_action
                                    )
                                    if order:
                                        self.positions[best_ticker] = {
                                            "side": best_action,
                                            "amount": amount,
                                            "entry_price": curr_px,
                                            "entry_timestamp": time.time(),
                                            "mfe": 0.0
                                        }
                                        self._save_state()
                                        msg = (f"{emoji} **[V35 최고확신도 진입]** {best_ticker} @ {curr_px:.4f}\n"
                                               f"{best_label} | 레버리지: {current_leverage}x (Dynamic) | 증거금: ${margin:.2f}")
                                        await self.telegram.send_message(msg)
                                        logger.info(f"🚀 [{best_ticker}] 사냥 개시! ({best_label}, {current_leverage}x)")
                    except Exception as e:
                        if "110007" in str(e) or "not enough" in str(e).lower():
                            logger.warning(f"💰 [Balance Guard] 진입 실패: 잔고 부족.")
                        else:
                            logger.error(f"🚨 [{best_ticker}] 주문 실패: {e}")

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"🛑 오류: {e}")
                await asyncio.sleep(30)

async def main():
    bot = V35LiveBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 V35 Live Stop.")
