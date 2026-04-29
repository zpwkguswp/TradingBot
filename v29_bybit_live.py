import argparse
import asyncio
import json
import logging
import math
import os
import time
import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from v30_train import CausalConv1d, TCN6LayerExtractor, FULL_UNIVERSE

from config import BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from exchange import ExchangeClient
from telegram_bot import TelegramBot
from v29_env import V29_Universal_Env
from v29_data_builder import run_builder

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# [V29 Phase 1] Multi-Coin Universal Alpha Live Engine (Phoenix Edition)
# ==============================================================================
MAX_POSITIONS = 10       # 🔥 [V31 동적 비중] 최대 보유 포지션 수
ALLOCATION_RATE = 0.10   # 🔥 [V31 동적 비중] 매 진입 시 총 잔고의 10% 사용
SWAP_THRESHOLD = 0.20    # 🔄 [V31 스왑 로직] 기존 시그널 대비 20% 우위 필요
SWAP_MIN_DIFF = 0.10     # 🔄 [V31 스왑 로직] 수수료 방어를 위한 절대적 시그널 차이 최소 0.10
LEVERAGE = 3
# 📊 [변동성 차등 레버리지] V29 Risk Parity
LEVERAGE_MAP = {
    # [Tier 1] 초우량 대장 (변동성 최하, 무빙 정직)
    'BTCUSDT': 10,  
    
    # [Tier 2] 메이저 플랫폼 코인 (시총 상위, 안정성 높음)
    'ETHUSDT': 5,   
    'BNBUSDT': 5,   
    'SOLUSDT': 5,   
    
    # [Tier 3] 준메이저 알트코인 (기본 레버리지 3배 방어)
    'XRPUSDT': 3,   
    'ADAUSDT': 3,
    'DOTUSDT': 3,
    'LTCUSDT': 3,
    'LINKUSDT': 3,
    'TRXUSDT': 3,
    'TONUSDT':3,
    'POLUST':3,
    'NEARUSDT': 3,
    'AVAXUSDT': 3,
    'UNIUSDT': 3,
    'APTUSDT': 3,
    
    # [Tier 4] 고위험 밈코인 (돌발 빔/꼬리 극심, 방어력 최대치)
    'DOGEUSDT': 2,  
}
TARGET_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "BNBUSDT", "DOGEUSDT", "DOTUSDT", "TRXUSDT", "TONUSDT", 
    "POLUST", "LTCUSDT", "NEARUSDT", "AVAXUSDT", "LINKUSDT", 
    "UNIUSDT", "APTUSDT"
]

# 🌐 [V30 Universal Alpha] 전체 유니버스 인덱스 맵 (v30_train.py와 완벽 일치 필수)
TIMEFRAME = "2h"
TF_HOURS = 2
PHASE_COLLECT_SEC_BEFORE = 5.0

# 🎯 [V29 최적 하이퍼파라미터 - Trial 59 적용 완료]
ADX_TH = 0.169327336793444
MAX_DISP = 0.0587715969881679
SL_ATR_COEF = 3.8350847145336115
FAR_TH = 2.2044178686597258
TRAIL_ACT = 0.020
PRESERVATION_RATIO = 0.70
TARGET_PROFIT = 0.008

DATA_DIR = "data_storage"
ELITE_DIR = "elite_weights"
MODEL_PATH = os.path.join(ELITE_DIR, f"v30_best_model_{TIMEFRAME}.zip")
STATE_FILE = "v30_live_state.json"
EXPERIENCE_LOG_FILE = "v30_experience_dataset.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# 📋 Experience Logger (V29 전체 라이프사이클 데이터셋 생성기) ============================
class ExperienceLogger:
    def __init__(self, filename=EXPERIENCE_LOG_FILE):
        self.filename = filename

    def log_experience(self, ticker, entry_time, exit_time, entry_price, exit_price, entry_obs, exit_obs, entry_action, pnl, mfe, mae, exit_reason):
        # Numpy 배열인 경우 리스트로 변환 (JSON 직렬화)
        if hasattr(entry_obs, "tolist"): entry_obs = entry_obs.tolist()
        if hasattr(exit_obs, "tolist"): exit_obs = exit_obs.tolist()

        data = {
            "ticker": ticker,
            "entry": {
                "timestamp": entry_time,
                "price": entry_price,
                "obs": entry_obs,
                "action": entry_action
            },
            "exit": {
                "timestamp": exit_time,
                "price": exit_price,
                "obs": exit_obs,
                "reason": exit_reason
            },
            "performance": {
                "pnl_pct": pnl,
                "mfe_pct": mfe,
                "mae_pct": mae
            },
            "recorded_at": datetime.now().isoformat()
        }

        logs = []
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "rb") as f:
                    logs = pickle.load(f)
            except Exception as e:
                logger.error(f"⚠️ 기존 데이터셋 로드 실패 (초기화): {e}")
                logs = []

        logs.append(data)

        try:
            with open(self.filename, "wb") as f:
                pickle.dump(logs, f)
            logger.info(f"✅ [ExperienceLogger] V29 Lifecycle 데이터 기록 완료: {ticker} ({pnl:.2f}%)")
        except Exception as e:
            logger.error(f"❌ [ExperienceLogger] 저장 실패: {e}")

# 🔧 Utils =====================================================================
def ticker_to_symbol(ticker: str) -> str:
    return f"{ticker[:-4]}/USDT:USDT" if ticker.endswith("USDT") else f"{ticker}/USDT:USDT"

def next_2h_close_unix() -> float:
    now = time.time()
    period = float(TF_HOURS * 60 * 60)
    return math.ceil(now / period) * period

async def _async_sleep_until(target_unix: float) -> None:
    d = target_unix - time.time()
    if d > 0:
        await asyncio.sleep(d)

class V29LiveBot:
    def __init__(self, is_dry_run: bool = False) -> None:
        self.is_dry_run = is_dry_run
        self.exchange = ExchangeClient(BYBIT_API_KEY, BYBIT_API_SECRET)
        self.telegram = TelegramBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, self.exchange, None)
        self.experience_logger = ExperienceLogger()
        self.positions: dict[str, dict] = {}
        self.is_running = True

        custom_objects = {"features_extractor_class": TCN6LayerExtractor}

        if not os.path.exists(MODEL_PATH):
            logger.warning(f"⚠️ 모델 파일 없음: {MODEL_PATH}")
            self.model = None
        else:
            # 🔄 [V30 Optimizer Error Bypass]
            # v30 훈련 시 Embedding에 별도 Weight Decay를 적용하여 2개의 Parameter Group이 생김.
            # SB3 기본 load는 1개의 Group만 예상하여 ValueError 발생 -> set_parameters로 우회 로드.
            try:
                self.model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
                logger.info(f"✅ V29 모델 로드 완료 (Standard): {MODEL_PATH}")
            except ValueError as e:
                if "parameter groups" in str(e):
                    logger.info("🔄 Optimizer mismatch detected. Loading weights only via set_parameters...")
                    
                    # 1. 임시 환경 생성 (관측 차원 148 = (36+1)*4)
                    import gymnasium as gym
                    class DummyEnv(gym.Env):
                        def __init__(self):
                            super().__init__()
                            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(148,), dtype=np.float32)
                            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                            self.metadata = {}
                        def reset(self, seed=None, options=None): return np.zeros(148, dtype=np.float32), {}
                        def step(self, action): return np.zeros(148, dtype=np.float32), 0.0, False, False, {}
                        def render(self): pass
                        def close(self): pass

                    # 2. 모델 초기화 (v30_train.py와 동일한 아키텍처 설정 필수)
                    policy_kwargs = dict(
                        features_extractor_class=TCN6LayerExtractor,
                        features_extractor_kwargs=dict(features_dim=256),
                        net_arch=dict(pi=[128, 64], vf=[256, 128]),
                    )
                    
                    self.model = PPO("MlpPolicy", DummyVecEnv([lambda: DummyEnv()]), 
                                     policy_kwargs=policy_kwargs, device="cpu")
                    
                    # 3. 가중치만 강제 로드 (Optimizer 파일인 policy.optimizer.pth를 무시하고 policy.pth만 로드)
                    import zipfile
                    import io
                    import torch
                    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
                        with archive.open("policy.pth") as f:
                            policy_weights = torch.load(io.BytesIO(f.read()), map_location="cpu")
                    
                    self.model.policy.load_state_dict(policy_weights)
                    logger.info(f"✅ V29 모델 로드 완료 (Weights Only - Direct): {MODEL_PATH}")
                else:
                    raise e

        self._load_state()

    def _load_state(self) -> None:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.positions = raw.get("positions", {})
            except Exception as e:
                logger.error("⚠️ 상태 로드 실패: %s", e)

    def _save_state(self) -> None:
        if self.is_dry_run: return
        payload = {"positions": self.positions, "updated_at": datetime.now(timezone.utc).isoformat()}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    async def fetch_latest_closure_info(self, ticker: str):
        if self.is_dry_run: return None
        symbol = ticker_to_symbol(ticker)
        try:
            trades = await self.exchange.exchange.fetch_my_trades(symbol, limit=10)
            if not trades: return None
            # 최신순 정렬
            trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)
            # Bybit v5: 'closedPnl' 또는 'realizedPnl' 필드 확인
            for t in trades:
                info = t.get('info', {})
                pnl_usd = float(info.get('closedPnl', info.get('realizedPnl', 0.0)))
                if abs(pnl_usd) > 0:
                    return {
                        "exit_price": float(t['price']),
                        "exit_time": t['datetime'],
                        "pnl_usd": pnl_usd,
                        "side": t['side']
                    }
            return None
        except Exception as e:
            logger.error(f"⚠️ [{ticker}] 최근 체결 조회 실패: {e}")
            return None

    async def sync_with_exchange(self):
        if self.is_dry_run: return
        try:
            # 🛡️ [10002 타임스탬프 에러 방어] CCXT 자체 자동 시간 보정 기능 활성화
            self.exchange.exchange.options['adjustForTimeDifference'] = True

            real_positions = await self.exchange.fetch_open_positions()

            # 🛡️ [유령 청산 방어 루프 차단]
            # API가 10002 에러로 실패했을 때, 빈 리스트([])를 반환하여 메모리를 지우는 상황 방어
            if len(real_positions) == 0 and len(self.positions) > 0:
                try:
                    # 간단한 잔고 조회로 API가 실제 살아있는지 교차 검증 (Cross-Check)
                    await self.exchange.exchange.fetch_balance()
                except Exception as e:
                    logger.warning(f"⚠️ API 불안정(10002 에러 등) 감지! 유령 청산을 막기 위해 메모리 상태를 보존합니다: {e}")
                    await self.exchange.exchange.load_time_difference()  # 시간 재동기화 시도
                    return  # 메모리 상태 지우지 않고 즉시 탈출

            real_symbols = [p['symbol'].replace('/USDT:USDT', 'USDT') for p in real_positions]

            for ticker in list(self.positions.keys()):
                if ticker not in real_symbols:
                    logger.info("🔔 [%s] 거래소에서의 포지션 종료 확인 (동기화 이후 처리)", ticker)

                    # 청산 정보 수집 시도
                    pos = self.positions.get(ticker)
                    closure = await self.fetch_latest_closure_info(ticker)

                    if closure:
                        exit_px = closure["exit_price"]
                        pnl_usd = closure["pnl_usd"]
                        pnl_pct = (pnl_usd / (pos.get("margin", 0.0) if pos.get("margin", 0.0) > 0 else (100.0))) * 100.0 # V31 margin fallback

                        msg = f"🔔 **[Phoenix Recovery]** {ticker} 포지션 청산 확인\n"
                        msg += f"청산시간: `{closure['exit_time']}`\n"
                        msg += f"진입가: `{pos['entry_price']:.4f}` -> 청산가: `{exit_px:.4f}`\n"
                        msg += f"손익률: `{pnl_pct:.2f}%` ($ {pnl_usd:.2f})"
                        await self.telegram.send_message(msg)

                        # 로그 기록 (pkl)
                        entry_obs = pos.get("entry_obs")
                        if entry_obs is None:
                            entry_obs = np.zeros(37).tolist()

                        self.experience_logger.log_experience(
                            ticker=ticker,
                            entry_time=pos.get("entry_timestamp", time.time()),
                            exit_time=time.time(),
                            entry_price=pos["entry_price"],
                            exit_price=exit_px,
                            entry_obs=entry_obs,
                            exit_obs=np.zeros(37).tolist(),
                            entry_action=pos.get("entry_action", 0.0),
                            pnl=pnl_pct,
                            mfe=pos.get("mfe", 0.0) * 100.0,
                            mae=pos.get("mae", 0.0) * 100.0,
                            exit_reason="External_Closure"
                        )
                    else:
                        await self.telegram.send_message(f"🔔 **[{ticker}]** 포지션 청산 확인 (체결 내역 조회 실패)")

                    del self.positions[ticker]

            for p in real_positions:
                ticker = p['symbol'].replace('/USDT:USDT', 'USDT')
                if ticker not in self.positions and ticker in TARGET_COINS:
                    # 🔄 [미등록 포지션 감지] MFE 초기값 계산 및 포지션 등록
                    entry_px = float(p['entryPrice'])
                    curr_px = float((await self.exchange.exchange.fetch_ticker(p['symbol']))['last'])
                    pnl_raw = (curr_px / entry_px - 1.0) if p['side'].lower() == "long" else (1.0 - curr_px / entry_px)
                    is_trail_active = True if pnl_raw >= TRAIL_ACT else False

                    logger.info(f"🔔 [{ticker}] 미등록 포지션 발견! (MFE 초기값: {pnl_raw*100:.2f}%)")
                    await self.telegram.send_message(f"🔔 **[포지션 자동등록]** {ticker}\n현재 손익률: `{pnl_raw*100:.2f}%`")

                    self.positions[ticker] = {
                        "side": p['side'].lower(),
                        "entry_price": entry_px,
                        "amount": float(p['contracts']),
                        "sl_price": float(p.get('stopLoss', 0.0)),
                        "trail_active": is_trail_active,
                        "mfe": max(0.0, pnl_raw),
                        "mae": min(0.0, pnl_raw),
                        "entry_obs": np.zeros(37).tolist(),
                        "entry_action": 0.0,
                        "entry_timestamp": time.time(),
                        "leverage": LEVERAGE
                    }
                    self._save_state()
            self._save_state()
        except Exception as e:
            logger.error("🛡️ 동기화 오류: %s", e)

    async def safe_market_close(self, ticker: str, exit_reason: str):
        pos = self.positions.get(ticker)
        if not pos: return

        curr_price = float((await self.exchange.exchange.fetch_ticker(ticker_to_symbol(ticker)))['last'])
        pnl_raw = (curr_price / pos["entry_price"] - 1.0) if pos["side"] == "long" else (1.0 - curr_price / pos["entry_price"])
        pnl_pct = pnl_raw * 100.0

        # 📸 [Exit Obs Capture] 청산 시점의 관측값 기록
        try:
            _, exit_obs = self.get_v29_observation(ticker)
        except Exception as e:
            logger.warning(f"⚠️ [{ticker}] 청산 관측값 캡처 실패: {e}")
            exit_obs = np.zeros(37).tolist()

        # 진입 정보 수집 (등록된 포지션 기반)
        entry_obs = pos.get("entry_obs")
        if entry_obs is None: entry_obs = np.zeros(37).tolist()
        entry_action = pos.get("entry_action", 0.0)
        entry_ts = pos.get("entry_timestamp", time.time())

        logger.info(f"📤 [{ticker}] 시장가 청산 시작. 이유: {exit_reason}, PnL: {pnl_pct:.2f}%")
        if not self.is_dry_run:
            try:
                # 🛡️ [중복 방지] 동일한 주문이 중복되지 않는 고유한 청산 주문 ID 생성 (110072 에러 방어)
                unique_exit_id = f"Close_{ticker}_{int(time.time() * 1000)}"

                await self.exchange.close_position_market(
                    symbol=ticker_to_symbol(ticker),
                    side="sell" if pos["side"] == "long" else "buy",
                    amount=pos["amount"],
                    client_id=unique_exit_id,
                    position_side=pos["side"]
                )
            except Exception as e:
                logger.error(f"❌ [{ticker}] 청산 실패 (API 오류): {e}")
                # 📌 거래소에서 실제로 청산됐는지 불명확하므로 메모리(상태 파일)에서 지우지 않고 다음 루프로 재시도
                return

        # ✅ [데이터 기록] 거래소 청산이 성공했음을 확인한 후에만 경험 데이터에 기록
        self.experience_logger.log_experience(
            ticker=ticker,
            entry_time=entry_ts,
            exit_time=time.time(),
            entry_price=pos["entry_price"],
            exit_price=curr_price,
            entry_obs=entry_obs,
            exit_obs=exit_obs,
            entry_action=entry_action,
            pnl=pnl_pct,
            mfe=pos.get("mfe", 0.0) * 100.0,
            mae=pos.get("mae", 0.0) * 100.0,
            exit_reason=exit_reason
        )

        realized_usd = pnl_raw * pos.get("margin", 0.0) # V31 동적 증거금 기반 실제 수익금
        alert_msg = f"📊 **[V29 청산]** {ticker}\n이유: `{exit_reason}`\n손익: `{pnl_pct:.2f}%` ($ {realized_usd:.2f})"
        await self.telegram.send_message(alert_msg)
        del self.positions[ticker]
        self._save_state()

    async def fetch_missing_5m_data(self, ticker: str):
        file_path = f"{DATA_DIR}/{ticker}_5m.parquet"
        if not os.path.exists(file_path): return False
        df = pd.read_parquet(file_path)
        last_ts = int(df['timestamp'].max())
        since = last_ts + 1
        new_data = []
        symbol = ticker_to_symbol(ticker)
        while True:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
                if not ohlcv or len(ohlcv) <= 1: break
                new_data.extend(ohlcv)
                since = int(ohlcv[-1][0]) + 1
                if len(ohlcv) < 500: break
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[{ticker}] 5m 데이터 동기화 오류: {e}")
                break

        if new_data:
            df_new = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_final = pd.concat([df, df_new]).drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
            df_final.to_parquet(file_path, index=False)
            logger.info(f"✅ {ticker} 5m 업데이트 완료 (+{len(df_new)})")
        return True

    def get_v29_observation(self, ticker: str):
        coin_id = FULL_UNIVERSE.index(ticker) if ticker in FULL_UNIVERSE else 0

        # 1. 래퍼(Wrapper) 껍데기 없이 순수 환경만 생성
        env = V29_Universal_Env(
            data_dir=DATA_DIR,
            coin_files=[f"{ticker}_{TIMEFRAME}.parquet"],
            coin_id=coin_id,
            split_type=None,
            target_profit=TARGET_PROFIT,
            far_th=FAR_TH,
            sl_atr_coef=SL_ATR_COEF,
            adx_th=ADX_TH,
            max_disp=MAX_DISP,
            trail_act=TRAIL_ACT
        )

        # 2. 내부 버퍼 초기화 및 첫 프레임 확보
        raw_obs = env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
            
        max_idx = len(env.df) - 1
        obs_history = [raw_obs]

        # 3. 🚨 [정속 주행 풀스캔] 
        # 건너뛰지 않고 0번부터 끝까지 1200번의 step을 직접 밟습니다.
        # 이렇게 해야 환경 내부의 지표(이동평균 등)가 실제 데이터로 생생하게 갱신됩니다!
        for _ in range(max_idx):
            # 🛡️ AI가 훈련 때 보던 정상적인 자금 스케일(10,000) 유지
            # 스텝 진행 중 파산(balance <= 0)하면 환경을 리셋하여 스케일 붕괴 방지
            cur_bal = getattr(env, 'balance', 10000.0)
            cur_nw = getattr(env, 'net_worth', 10000.0)
            if cur_bal <= 0 or cur_nw <= 0:
                if hasattr(env, 'balance'): env.balance = 10000.0
                if hasattr(env, 'net_worth'): env.net_worth = 10000.0
                if hasattr(env, 'max_net_worth'): env.max_net_worth = 10000.0
                if hasattr(env, 'initial_balance'): env.initial_balance = 10000.0
            
            res = env.step(np.array([0.0]))
            raw_obs = res[0]
            obs_history.append(raw_obs)

        # 4. 가장 마지막에 쌓인 '진짜 최신' 4개의 프레임 추출
        last_4_raw = obs_history[-4:]
        
        # 5. 차원 붕괴 방지 및 37차원 규격 수동 조립
        frames = []
        for obs in last_4_raw:
            new_obs = np.zeros(37, dtype=np.float32)
            obs_1d = np.nan_to_num(np.array(obs).flatten())
            copy_len = min(len(obs_1d), 36)
            new_obs[:copy_len] = obs_1d[:copy_len]
            new_obs[36] = float(coin_id)
            frames.append(new_obs)

        # 6. 모델 입력 규격 (1, 148) 으로 압축 완료
        stacked_obs = np.concatenate(frames).reshape(1, -1)
        single_obs = frames[-1].copy()

        return stacked_obs, single_obs

    async def run(self) -> None:
        logger.info(f"🚀 V30 Universal Alpha Engine Starting... (Target: {len(TARGET_COINS)} coins)")

        # 🔁 [시간 동기화 재시도 처리] 시작 시 최대 5번까지 재시도
        sync_success = False
        for attempt in range(5):
            try:
                # CCXT 자체 시간 동기화 함수 강제 호출
                await self.exchange.exchange.load_time_difference()
                logger.info("🔁 [Exchange] 바이비트 서버 시간 동기화 성공 (10002 에러 방어선 구축)")
                sync_success = True
                break
            except Exception as e:
                logger.warning(f"⚠️ 시간 동기화 실패 ({attempt+1}/5) - 3초 후 재시도.. 원인: {e}")
                await asyncio.sleep(3)

        # 5번 모두 실패하면 서버와의 통신 상태를 점검해야 하므로 즉시 종료
        if not sync_success:
            logger.error("🛑 5번 재시도에도 시간 동기화 실패! 서버와의 연결 상태를 점검하세요. 종료합니다.")
            return

        await self.sync_with_exchange()

        while self.is_running:
            try:
                next_close = next_2h_close_unix()

                # ==========================================================
                # ⚡ Fast Loop (15초 간격)
                # ==========================================================
                while time.time() < next_close - PHASE_COLLECT_SEC_BEFORE:

                    # 🛡️ [이중 방어 장치 1] 매 15초마다 바이비트 실제 포지션과 메모리 강제 동기화
                    # 메모리의 상태(self.positions)가 비어도 바이비트 포지션이 있으면 업데이트합니다.
                    await self.sync_with_exchange()

                    if self.positions:
                        tickers = await self.exchange.exchange.fetch_tickers([ticker_to_symbol(t) for t in self.positions.keys()])
                        for ticker, pos in list(self.positions.items()):
                            tick = tickers.get(ticker_to_symbol(ticker))
                            if not tick: continue
                            curr_price = float(tick['last'])

                            # SL 체크
                            if (pos["side"] == "long" and curr_price <= pos["sl_price"]) or \
                               (pos["side"] == "short" and curr_price >= pos["sl_price"]):
                                await self.safe_market_close(ticker, "Soft_SL")
                                continue

                            # Trailing Stop 체크
                            pnl_raw = (curr_price / pos["entry_price"] - 1.0) if pos["side"] == "long" else (1.0 - curr_price / pos["entry_price"])
                            if pnl_raw > pos["mfe"]: pos["mfe"] = pnl_raw
                            if pnl_raw < pos.get("mae", 0.0): pos["mae"] = pnl_raw

                            if pos["mfe"] >= TRAIL_ACT:
                                if not pos.get("trail_active"):
                                    pos["trail_active"] = True
                                    logger.info(f"📌 [{ticker}] Trailing Stop 활성화!")
                                if pnl_raw < pos["mfe"] * PRESERVATION_RATIO:
                                    await self.safe_market_close(ticker, "Trailing_Stop")
                                    continue

                            # ⏳ [Time Stop] 진입 후 24시간(86400초) 경과했는데 손익이 1% 미만이면 손실 없이 청산
                            time_elapsed = time.time() - pos.get("entry_timestamp", time.time())
                            pnl_pct = pnl_raw * 100.0

                            if time_elapsed > 86400 and pnl_pct < 1.0:
                                logger.info(f"⏳ [{ticker}] 시간초과 발동! (24시간 경과, 현재 손익률: {pnl_pct:.2f}%, 수익 미달)")
                                await self.safe_market_close(ticker, "Time_Stop")
                                continue

                    await asyncio.sleep(15)

                # ==========================================================
                # 🌙 Slow Loop (2H 봉 마감 후 신호 계산)
                # ==========================================================
                logger.info("🌙 [2H Boundary] 새 캔들 데이터 동기화 및 신호 분석 시작")

                # 🛡️ [시간 드리프트 방어 장치] 신호 계산 및 진입 직전에 시점 재맞춤!
                try:
                    await self.exchange.exchange.load_time_difference()
                except Exception as e:
                    logger.warning(f"⚠️ 2H 봉마감 시간 동기화 실패 (무시하고 진행): {e}")

                # 🛡️ [이중 방어 장치 2] 새로운 진입이 결정되기 직전, 바이비트 포지션과 최종 동기화!
                await self.sync_with_exchange()

                for t in TARGET_COINS:
                    await self.fetch_missing_5m_data(t)
                    run_builder(t)

                if not self.model:
                    custom_objects = {"features_extractor_class": TCN6LayerExtractor}
                    self.model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects) if os.path.exists(MODEL_PATH) else None

                if self.model:
                    candidate_signals = []
                    current_signals = {}  # 🌟 [V31 추가] 현재 들고 있는 포지션의 예측값 저장용

                    # 1️⃣ Phase 1: 모든 코인 관측값 생성 및 반전 청산 (Signal Flip)
                    for ticker in TARGET_COINS:
                        try:
                            stacked_obs, single_obs = self.get_v29_observation(ticker)

                            # 🔬 [V29 임베딩 민감도 검증 테스트] (최초 1회만 실행)
                            if ticker == TARGET_COINS[0] and not hasattr(self, '_embedding_tested'):
                                self._embedding_tested = True
                                logger.info(f"🔬 [Embedding Test] Base Ticker: {ticker} (Checking Sensitivity)")
                                for cid in range(10):
                                    test_stacked = stacked_obs.copy()
                                    for f_idx in range(4):
                                        test_stacked[0, (f_idx + 1) * 37 - 1] = float(cid)
                                    t_action, _ = self.model.predict(test_stacked, deterministic=True)
                                    logger.info(f"🔬 [Embedding Test] Coin ID {cid} -> Prediction: {float(t_action[0]):.4f}")

                            action, _ = self.model.predict(stacked_obs, deterministic=True)
                            act_val = float(action[0])

                            signal = "WAIT"
                            if act_val > 0.05: signal = "LONG"
                            elif act_val < -0.05: signal = "SHORT"

                            logger.info(f"🤖 [{ticker}] Prediction: {act_val:.4f} -> {signal}")

                            # [반전 청산] 반대 방향 신호 발생 시 청산 (방향 전환 정보)
                            if ticker in self.positions:
                                pos = self.positions[ticker]
                                if (pos["side"] == "long" and signal == "SHORT") or (pos["side"] == "short" and signal == "LONG"):
                                    await self.safe_market_close(ticker, "Signal_Flip")
                                else:
                                    current_signals[ticker] = act_val # 살아남은 기존 포지션의 현재 시그널 저장

                            # [신호 수집] 반전 청산 안 당한 코인 중 강한 진입 신호를 후보로 추가
                            if abs(act_val) > 0.95 and ticker not in self.positions:
                                candidate_signals.append({
                                    "ticker": ticker,
                                    "act_val": act_val,
                                    "abs_val": abs(act_val),
                                    "obs": single_obs,
                                    "stacked_obs": stacked_obs
                                })
                                logger.info(f"🎯 [신호 확보] {ticker}: {act_val:.4f}")

                        except Exception as e:
                            logger.error(f"[{ticker}] 신호 분석 및 관측값 생성 오류: {e}")

                    # 2️⃣ Phase 2: [우선순위 정렬 및 스왑(Swap) 검증]
                    candidate_signals.sort(key=lambda x: -x["abs_val"]) # 강한 순 (내림차순)
                    
                    holdings_list = []
                    for t, a_val in current_signals.items():
                        holdings_list.append({"ticker": t, "abs_val": abs(a_val)})
                    holdings_list.sort(key=lambda x: x["abs_val"]) # 약한 순 (오름차순)

                    final_entry_targets = []
                    
                    for candidate in candidate_signals:
                        if len(self.positions) + len(final_entry_targets) < MAX_POSITIONS:
                            final_entry_targets.append(candidate)
                        else:
                            # 꽉 찼으면 스왑 조건 검사
                            if holdings_list:
                                weakest = holdings_list[0]
                                cand_abs = candidate["abs_val"]
                                weak_abs = weakest["abs_val"]
                                
                                # 🔥 [이식 완료 1] 스왑 쿨타임 (최소 4캔들 = 8시간 방어)
                                weakest_pos = self.positions[weakest['ticker']]
                                time_held = time.time() - weakest_pos.get("entry_timestamp", time.time())
                                is_cooled_down = time_held > (4 * 2 * 3600) # 4캔들 * 2시간 * 3600초
                                
                                # 🔥 [이식 완료 2] 1.0 포화 상태 무한 스왑 방지 (Tie-breaker)
                                is_saturated = (cand_abs >= 0.95 and weak_abs >= 0.95)

                                # 🌟 [V31 스왑 최종 조건] 쿨타임 충족 & 포화상태 아닐 때만 스왑 진행
                                if is_cooled_down and not is_saturated:
                                    # 조건 1.2배 초과 AND 절대 차이 0.10 초과
                                    if cand_abs > weak_abs * (1.0 + SWAP_THRESHOLD) and (cand_abs - weak_abs) > SWAP_MIN_DIFF:
                                        logger.info(f"🔄 [Swap] 더 강력한 시그널 발견! {weakest['ticker']}({weak_abs:.4f}) -> {candidate['ticker']}({cand_abs:.4f})")
                                        await self.safe_market_close(weakest['ticker'], "Position_Swap")
                                        holdings_list.pop(0) # 가장 약한 놈 청산했으니 리스트에서 제거
                                        final_entry_targets.append(candidate) # 새 후보 등록
                                    else:
                                        break # 가장 강한 후보가 못 이기면 다음 후보도 못 이김
                                else:
                                    break # 쿨타임 대기 중이거나 둘 다 0.95 이상 100% 확신 상태면 스왑 생략
                            else:
                                break

                    # 3️⃣ Phase 3: 순서대로 실제 진입 (동적 비중 조절)
                    if final_entry_targets:
                        try:
                            total_equity = await self.exchange.get_total_equity()
                        except Exception as e:
                            logger.error(f"⚠️ 총 잔고 조회 실패: {e}")
                            total_equity = 100.0 # 에러 방지용 Fallback
                            
                        for idx, target in enumerate(final_entry_targets):
                            ticker = target["ticker"]
                            act_val = target["act_val"]
                            signal = "LONG" if act_val > 0.2 else "SHORT"

                            logger.info(f"🚀 [실제 진입 {idx+1}/{len(final_entry_targets)}] {ticker} 주문 발행 (값: {act_val:.4f})")

                            try:
                                target_lev = LEVERAGE_MAP.get(ticker, LEVERAGE)
                                curr_tick = await self.exchange.exchange.fetch_ticker(ticker_to_symbol(ticker))
                                curr_price = float(curr_tick['last'])

                                df_2h = pd.read_parquet(f"{DATA_DIR}/{ticker}_2h.parquet")
                                h1_atr = float(df_2h.iloc[-1].get("atr_raw", 0.002)) * curr_price
                                sl_dist = SL_ATR_COEF * h1_atr
                                sl_px = (curr_price - sl_dist) if signal == "LONG" else (curr_price + sl_dist)

                                # 🌟 [V31 동적 비중 조절]
                                trade_margin_usdt = total_equity * ALLOCATION_RATE
                                base_amount = (trade_margin_usdt * target_lev) / curr_price
                                
                                try:
                                    min_amount = await self.exchange.fetch_market_min_amount(ticker_to_symbol(ticker))
                                except:
                                    min_amount = 0.0

                                final_amount = max(base_amount, min_amount)
                                req_margin = (final_amount * curr_price) / target_lev

                                if req_margin > (trade_margin_usdt * 1.5):
                                    abort_msg = f"⚠️ **[{ticker} 진입 취소]** 최소 수량({min_amount}) 초과로 필요증거금이 ${req_margin:.2f}로 한도(${trade_margin_usdt:.2f}) 초과"
                                    logger.warning(abort_msg)
                                    await self.telegram.send_message(abort_msg)
                                    continue

                                if not self.is_dry_run:
                                    unique_client_id = f"V31_{ticker}_{int(time.time() * 1000)}"
                                    await self.exchange.place_order_with_sl(
                                        symbol=ticker_to_symbol(ticker),
                                        side="buy" if signal == "LONG" else "sell",
                                        amount=final_amount,
                                        client_id=unique_client_id,
                                        leverage=target_lev,
                                        sl_price=sl_px, position_side=signal.lower()
                                    )

                                self.positions[ticker] = {
                                    "side": signal.lower(), "entry_price": curr_price, "amount": final_amount,
                                    "sl_price": sl_px, "trail_active": False, "mfe": 0.0, "mae": 0.0,
                                    "leverage": target_lev,
                                    "margin": trade_margin_usdt, # 수익률 계산을 위해 증거금 저장
                                    "entry_obs": target["obs"].tolist(),
                                    "entry_action": act_val,
                                    "entry_timestamp": time.time()
                                }
                                self._save_state()
                                logger.info(f"✅ [{ticker}] {signal} 진입 완료 (SL: {sl_px:.4f}, 증거금: ${trade_margin_usdt:.2f})")
                                await self.telegram.send_message(f"🟢 **[V31 진입]** {ticker} {signal} @ {curr_price:.4f}\n💰 증거금: ${trade_margin_usdt:.2f} ({ALLOCATION_RATE*100}%)")

                                if idx < len(final_entry_targets) - 1:
                                    term_sec = 3
                                    logger.info(f"⏳ 다음 코인 진입 전 {term_sec}초간 대기 중..")
                                    await asyncio.sleep(term_sec)

                            except Exception as e:
                                logger.error(f"[{ticker}] 실제 진입 주문 실행 오류: {e}")

                await _async_sleep_until(next_2h_close_unix() + 5.0)
            except Exception as e:
                logger.error(f"🛑 메인 루프 치명적 오류: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    async def main():
        bot = V29LiveBot(is_dry_run=args.dry_run)
        await bot.run()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Live Stop.")
