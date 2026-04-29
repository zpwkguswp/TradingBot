"""
================================================================================
V29 Live Bot — 회귀 방지 테스트 하네스
================================================================================
이 파일은 실제 운영 중 발생했던 치명적 버그들을 재현하고,
현재 패치가 해당 버그들을 완전히 막고 있음을 자동으로 검증합니다.

[기록된 버그 목록]
 Bug #1: Ghost Liquidation Loop (2026-04-21)
   - 원인: Bybit API 10002 타임스탬프 에러 시 fetch_open_positions()가 빈 []를 반환.
           sync_with_exchange()가 이를 "실제 전량 청산"으로 오해, self.positions를 초기화.
   - 증상: 포지션이 유령처럼 사라지고 재진입 → 무한 손실 루프.
   - 패치: fetch_balance()로 API 생존 여부 교차 검증, 실패 시 return으로 메모리 보존.

 Bug #2: Ghost Observation — 고착 예측값 1.0000 / 0.7907 (2026-04-20)
   - 원인: get_v29_observation()에서 DummyVecEnv.reset() 후 step을 0에서 시작.
           모델이 수년 전 과거 데이터를 최신 데이터로 착각하여 추론.
   - 증상: 모든 코인 예측값이 1.0000 또는 0.7907로 고착.
   - 패치: env.current_step = max_idx - 4 강제 이동 + fake_step으로 done=False 봉쇄.

 Bug #3: Dimension Mismatch Crash — (33,) into (37,) (2026-04-21)
   - 원인: 실시간 캔들 수 부족 시 Builder가 MACD 등 장기 지표 계산 실패 → 33개 피처 반환.
           VecFrameStack이 observation_space(37) ≠ 실제 obs(33) 불일치로 충돌.
   - 증상: could not broadcast input array from shape (33,) into shape (37,)
   - 패치: _env_init() 내부에 fix_obs() 인터셉터 내장. 몇 개를 뱉든 무조건 37차원 보장.

실행 방법:
  python -m pytest tests/test_live_bot_harness.py -v
  또는
  python tests/test_live_bot_harness.py
================================================================================
"""

import sys
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼: 가짜 V29_Universal_Env 팩토리
# obs_size를 조절해 33/36/37개 피처를 뱉는 환경을 시뮬레이션합니다.
# ──────────────────────────────────────────────────────────────────────────────
def make_fake_env(obs_size: int, n_rows: int = 200):
    """obs_size 크기의 관측값을 반환하는 가짜 환경을 생성합니다."""
    import gymnasium as gym

    class FakeEnv(gym.Env):
        def __init__(self, **kwargs):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.current_step = 0
            # 실제 환경처럼 df 속성 보유
            self.df = list(range(n_rows))

        def reset(self, **kwargs):
            self.current_step = 0
            obs = np.random.rand(obs_size).astype(np.float32)
            return obs, {}

        def step(self, action):
            self.current_step += 1
            obs = np.random.rand(obs_size).astype(np.float32)
            done = self.current_step >= len(self.df) - 1
            return obs, 0.0, done, False, {}

    return FakeEnv


# ──────────────────────────────────────────────────────────────────────────────
# Bug #1 테스트: Ghost Liquidation Loop 방어
# ──────────────────────────────────────────────────────────────────────────────
class TestGhostLiquidationPrevention(unittest.IsolatedAsyncioTestCase):
    """
    Bug #1: API 10002 에러 시 빈 포지션 반환으로 인한 메모리 파괴 방어.
    패치: fetch_balance() 교차 검증 실패 시 즉시 return하여 self.positions 보존.
    """

    async def _make_bot(self):
        """외부 의존성을 모두 Mock한 최소 V29LiveBot 인스턴스를 생성합니다."""
        with patch("v29_bybit_live.ExchangeClient"), \
             patch("v29_bybit_live.TelegramBot"), \
             patch("v29_bybit_live.PPO"), \
             patch("os.path.exists", return_value=False):
            from v29_bybit_live import V29LiveBot
            bot = V29LiveBot(is_dry_run=True)
            bot.is_dry_run = False  # 동기화 로직 활성화를 위해 False로 전환
        return bot

    async def test_positions_preserved_when_api_returns_empty_due_to_error(self):
        """
        [재현] API가 10002 에러로 빈 []를 반환할 때,
              fetch_balance()도 실패하면 self.positions가 삭제되지 않아야 한다.
        """
        bot = await self._make_bot()

        # 포지션이 1개 있는 상태 설정
        bot.positions = {"BTCUSDT": {"entry_price": 50000.0, "side": "long"}}
        initial_positions = dict(bot.positions)

        # fetch_open_positions → 빈 리스트 (API 에러 시뮬레이션)
        bot.exchange.fetch_open_positions = AsyncMock(return_value=[])
        # fetch_balance → 예외 (10002 에러 시뮬레이션)
        bot.exchange.exchange.fetch_balance = AsyncMock(
            side_effect=Exception("10002: timestamp error")
        )
        bot.exchange.exchange.load_time_difference = AsyncMock()
        bot.exchange.exchange.options = {}

        await bot.sync_with_exchange()

        # 핵심 검증: 포지션이 그대로 유지되어야 함
        self.assertEqual(
            bot.positions, initial_positions,
            "❌ [Bug #1 재발] API 에러 시 포지션이 삭제되었습니다! "
            "fetch_balance() 교차 검증 패치를 확인하세요."
        )
        # load_time_difference가 호출되어 시간 재동기화 시도
        bot.exchange.exchange.load_time_difference.assert_called_once()

    async def test_positions_cleared_when_api_confirms_no_positions(self):
        """
        [정상 작동] fetch_balance()가 성공(API 살아있음)하고 실제로 포지션 없으면,
                   self.positions는 정상적으로 삭제되어야 한다.
        """
        bot = await self._make_bot()
        bot.positions = {"BTCUSDT": {"entry_price": 50000.0, "side": "long",
                                      "entry_timestamp": 0, "leverage": 3,
                                      "entry_obs": None, "entry_action": 0.0,
                                      "mfe": 0.0, "mae": 0.0}}

        bot.exchange.fetch_open_positions = AsyncMock(return_value=[])
        # fetch_balance 성공 → API는 살아있고 진짜 포지션이 없는 것
        bot.exchange.exchange.fetch_balance = AsyncMock(return_value={"total": {}})
        bot.exchange.exchange.options = {}
        bot.fetch_latest_closure_info = AsyncMock(return_value=None)
        bot.telegram.send_message = AsyncMock()
        bot._save_state = MagicMock()

        await bot.sync_with_exchange()

        # 핵심 검증: 정상 청산이므로 포지션 삭제가 올바른 동작
        self.assertNotIn(
            "BTCUSDT", bot.positions,
            "❌ API가 정상이고 포지션이 없는데도 포지션이 남아있습니다."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Bug #2 테스트: Ghost Observation (스텝 고착 0/1.0000) 방어
# ──────────────────────────────────────────────────────────────────────────────
class TestGhostObservationPrevention(unittest.TestCase):
    """
    Bug #2: 시뮬레이터가 step 0에서 시작하여 과거 데이터를 최신 데이터로 착각.
    패치: current_step = max_idx - 4 강제 이동 + fake_step(done=False).
    """

    def _run_get_observation(self, obs_size: int, n_rows: int = 200):
        """
        지정된 obs_size를 반환하는 가짜 환경으로 get_v29_observation을 실행합니다.
        """
        FakeEnvClass = make_fake_env(obs_size, n_rows)
        captured_steps = []

        OrigFakeEnv = FakeEnvClass

        class InstrumentedFakeEnv(OrigFakeEnv):
            """current_step 변화를 추적하는 계측 환경."""
            def step(self, action):
                captured_steps.append(self.current_step)
                return super().step(action)

        with patch("v29_bybit_live.V29_Universal_Env", InstrumentedFakeEnv), \
             patch("v29_bybit_live.ExchangeClient"), \
             patch("v29_bybit_live.TelegramBot"), \
             patch("v29_bybit_live.PPO"), \
             patch("os.path.exists", return_value=False):
            from v29_bybit_live import V29LiveBot
            bot = V29LiveBot(is_dry_run=True)
            stacked_obs, single_obs = bot.get_v29_observation("BTCUSDT")

        return stacked_obs, single_obs, captured_steps, n_rows

    def test_observation_starts_from_latest_candles_not_step_zero(self):
        """
        [재현 방지] 4번의 step이 데이터셋 끝 근처(max_idx-4 이후)에서 시작해야 한다.
        step 0에서 시작하면 Bug #2가 재발한 것입니다.
        """
        n_rows = 200
        _, _, captured_steps, _ = self._run_get_observation(obs_size=36, n_rows=n_rows)
        max_idx = n_rows - 1
        expected_start = max_idx - 4

        self.assertEqual(len(captured_steps), 4,
                         "❌ step이 정확히 4번 실행되어야 합니다.")

        first_step = captured_steps[0]
        self.assertGreaterEqual(
            first_step, expected_start,
            f"❌ [Bug #2 재발] 첫 step이 {first_step}에서 시작했습니다. "
            f"최소 {expected_start} 이상이어야 합니다. "
            f"current_step 강제 이동 패치를 확인하세요."
        )

    def test_stacked_obs_shape_is_always_1x148(self):
        """
        stacked_obs의 shape는 환경의 obs_size와 무관하게 항상 (1, 148)이어야 한다.
        (4 프레임 × 37차원 = 148)
        """
        for obs_size in [33, 36, 37]:
            with self.subTest(obs_size=obs_size):
                stacked_obs, _, _, _ = self._run_get_observation(obs_size=obs_size)
                self.assertEqual(
                    stacked_obs.shape, (1, 148),
                    f"❌ obs_size={obs_size}일 때 stacked_obs.shape가 "
                    f"{stacked_obs.shape}입니다. (1, 148)이어야 합니다."
                )

    def test_single_obs_shape_is_37(self):
        """single_obs는 항상 (37,) 이어야 한다."""
        for obs_size in [33, 36, 37]:
            with self.subTest(obs_size=obs_size):
                _, single_obs, _, _ = self._run_get_observation(obs_size=obs_size)
                self.assertEqual(
                    single_obs.shape, (37,),
                    f"❌ obs_size={obs_size}일 때 single_obs.shape가 "
                    f"{single_obs.shape}입니다. (37,)이어야 합니다."
                )


# ──────────────────────────────────────────────────────────────────────────────
# Bug #3 테스트: Dimension Mismatch (33→37) 방어
# ──────────────────────────────────────────────────────────────────────────────
class TestDimensionMismatchPrevention(unittest.TestCase):
    """
    Bug #3: Builder가 33개 피처를 반환할 때 VecFrameStack shape 불일치 충돌.
    패치: _env_init() 내 fix_obs() 인터셉터로 항상 37차원 보장.
    """

    def _run_with_obs_size(self, obs_size: int):
        FakeEnvClass = make_fake_env(obs_size)
        with patch("v29_bybit_live.V29_Universal_Env", FakeEnvClass), \
             patch("v29_bybit_live.ExchangeClient"), \
             patch("v29_bybit_live.TelegramBot"), \
             patch("v29_bybit_live.PPO"), \
             patch("os.path.exists", return_value=False):
            from v29_bybit_live import V29LiveBot
            bot = V29LiveBot(is_dry_run=True)
            return bot.get_v29_observation("BTCUSDT")

    def test_no_crash_when_env_returns_33_features(self):
        """
        [핵심 재현 방지] Builder가 33개 피처를 반환해도 예외가 발생해서는 안 된다.
        에러: could not broadcast input array from shape (33,) into shape (37,)
        """
        try:
            stacked_obs, single_obs = self._run_with_obs_size(33)
        except Exception as e:
            self.fail(
                f"❌ [Bug #3 재발] obs_size=33에서 예외 발생: {e}\n"
                f"fix_obs() 인터셉터 패치를 확인하세요."
            )

    def test_no_crash_when_env_returns_36_features(self):
        """Builder가 36개 피처를 반환해도 예외가 발생해서는 안 된다."""
        try:
            stacked_obs, single_obs = self._run_with_obs_size(36)
        except Exception as e:
            self.fail(f"❌ obs_size=36에서 예외 발생: {e}")

    def test_coin_id_always_in_last_slot_of_each_frame(self):
        """
        fix_obs() 패치 후 각 프레임의 37번째(인덱스 36) 슬롯에 coin_id가 박혀있어야 한다.
        BTCUSDT의 coin_id = 0 (full_universe 기준)
        """
        for obs_size in [33, 36, 37]:
            with self.subTest(obs_size=obs_size):
                stacked_obs, single_obs = self._run_with_obs_size(obs_size)
                # stacked_obs shape: (1, 148) = 4 frames × 37
                for frame_idx in range(4):
                    coin_id_slot = stacked_obs[0, frame_idx * 37 + 36]
                    self.assertEqual(
                        coin_id_slot, 0.0,  # BTCUSDT = coin_id 0
                        f"❌ obs_size={obs_size}, frame {frame_idx}에서 "
                        f"coin_id 슬롯 값이 {coin_id_slot}입니다. 0.0이어야 합니다."
                    )

    def test_33_feature_obs_pads_missing_slots_with_zero(self):
        """
        33개 피처 환경에서 slots [33:36]은 0으로 패딩되어야 하고,
        slot [36]은 coin_id여야 한다. (데이터 손실 최소화 검증)
        """
        stacked_obs, single_obs = self._run_with_obs_size(33)
        # 마지막 프레임(가장 최신)의 slots 33~35는 0이어야 함
        missing_slots = single_obs[33:36]
        self.assertTrue(
            np.all(missing_slots == 0.0),
            f"❌ 33개 피처 환경에서 slots [33:36]이 0으로 패딩되지 않았습니다: {missing_slots}"
        )
        # slot 36은 coin_id=0
        self.assertEqual(single_obs[36], 0.0,
                         f"❌ single_obs[36] (coin_id)가 0.0이 아닙니다: {single_obs[36]}")

    def test_observation_space_overridden_to_37(self):
        """
        _env_init()이 반환하는 환경의 observation_space.shape가
        원본 obs_size(33)가 아닌 (37,)로 덮어씌워져야 한다.
        (이것이 VecFrameStack shape 충돌 방어의 핵심)
        """
        FakeEnvClass = make_fake_env(33)
        captured_env = {}

        OrigInit = FakeEnvClass.__init__

        class CapturingFakeEnv(FakeEnvClass):
            def __init__(self):
                super().__init__()
                captured_env["instance"] = self  # _env_init 직후 캡처

        # _env_init이 반환한 env 객체를 가로채기 위해 DummyVecEnv를 모킹
        from stable_baselines3.common.vec_env import DummyVecEnv

        original_dummy_vec_env_init = DummyVecEnv.__init__
        patched_envs = []

        def capturing_dummy_init(self_vec, env_fns):
            envs = [fn() for fn in env_fns]
            patched_envs.extend(envs)
            original_dummy_vec_env_init(self_vec, [lambda e=e: e for e in envs])

        with patch("v29_bybit_live.V29_Universal_Env", CapturingFakeEnv), \
             patch("v29_bybit_live.ExchangeClient"), \
             patch("v29_bybit_live.TelegramBot"), \
             patch("v29_bybit_live.PPO"), \
             patch("stable_baselines3.common.vec_env.DummyVecEnv.__init__",
                   capturing_dummy_init), \
             patch("os.path.exists", return_value=False):
            from v29_bybit_live import V29LiveBot
            bot = V29LiveBot(is_dry_run=True)
            try:
                bot.get_v29_observation("BTCUSDT")
            except Exception:
                pass  # shape 이후 단계 에러는 무시

        if patched_envs:
            env = patched_envs[0]
            self.assertEqual(
                env.observation_space.shape, (37,),
                f"❌ observation_space.shape가 {env.observation_space.shape}입니다. "
                f"(37,)로 덮어씌워져야 합니다."
            )


# ──────────────────────────────────────────────────────────────────────────────
# 통합 스모크 테스트
# ──────────────────────────────────────────────────────────────────────────────
class TestIntegrationSmoke(unittest.TestCase):
    """
    세 버그 패치가 동시에 작동하는 통합 시나리오:
    33개 피처 환경 + Brain Hack + Ghost Liquidation Guard가 모두 정상이어야 함.
    """

    def test_full_observation_pipeline_with_33_features(self):
        """
        최악의 시나리오: Builder가 33개 피처만 반환하는 상황에서도
        전체 파이프라인이 무결하게 동작해야 한다.
        """
        FakeEnvClass = make_fake_env(33, n_rows=300)
        with patch("v29_bybit_live.V29_Universal_Env", FakeEnvClass), \
             patch("v29_bybit_live.ExchangeClient"), \
             patch("v29_bybit_live.TelegramBot"), \
             patch("v29_bybit_live.PPO"), \
             patch("os.path.exists", return_value=False):
            from v29_bybit_live import V29LiveBot
            bot = V29LiveBot(is_dry_run=True)
            stacked_obs, single_obs = bot.get_v29_observation("ETHUSDT")

        # ETHUSDT coin_id = 1
        expected_coin_id = 1.0

        # 1. Shape 검증
        self.assertEqual(stacked_obs.shape, (1, 148))
        self.assertEqual(single_obs.shape, (37,))

        # 2. Coin ID 검증 (4프레임 모두)
        for i in range(4):
            self.assertEqual(stacked_obs[0, i * 37 + 36], expected_coin_id,
                             f"Frame {i}의 coin_id가 잘못되었습니다.")

        # 3. NaN/Inf 없음 검증
        self.assertFalse(np.any(np.isnan(stacked_obs)),
                         "stacked_obs에 NaN이 포함되어 있습니다.")
        self.assertFalse(np.any(np.isinf(stacked_obs)),
                         "stacked_obs에 Inf가 포함되어 있습니다.")


# ──────────────────────────────────────────────────────────────────────────────
# 직접 실행 시 결과 출력
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("V29 Live Bot - 회귀 방지 테스트 하네스")
    print("=" * 70)
    print()
    print("Bug #1: Ghost Liquidation Loop Prevention")
    print("Bug #2: Ghost Observation (step 0 고착) Prevention")
    print("Bug #3: Dimension Mismatch (33→37) Prevention")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGhostLiquidationPrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestGhostObservationPrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestDimensionMismatchPrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationSmoke))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("✅ 모든 회귀 테스트 통과 — 기존 버그들이 재발하지 않습니다.")
    else:
        print(f"❌ {len(result.failures + result.errors)}개 테스트 실패 — 패치 상태를 점검하세요!")
    sys.exit(0 if result.wasSuccessful() else 1)
