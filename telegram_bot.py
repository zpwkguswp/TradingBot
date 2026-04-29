"""
==============================================================================
  텔레그램 알림 모듈 (telegram_bot.py) - V13.8.9
  - Feature: aiohttp 기반 순수 API 통신 (추가 라이브러리 설치 불필요)
  - Fix: Markdown 문법 충돌로 인한 400 에러 원천 차단
==============================================================================
"""

import logging
import aiohttp

class TelegramBot:
    def __init__(self, token, chat_id, exchange, db):
        self.token = token
        self.chat_id = chat_id
        self.exchange = exchange
        self.db = db
        self.logger = logging.getLogger(__name__)
        
        # 텔레그램 API 주소
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = None
        self.session = aiohttp.ClientSession()
        
        self.logger.info("📡 순정 텔레그램 통신망 연결 준비 완료.")

    async def send_message(self, text):
        """특수기호 400 에러 걱정 없는 순수 텍스트 전송"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text
            # parse_mode를 아예 빼버려서 400 에러를 완벽히 차단합니다.
        }
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    self.logger.error(f"❌ 텔레그램 전송 실패 (상태코드: {response.status}): {error_msg}")
        except Exception as e:
            self.logger.error(f"❌ 텔레그램 통신 에러: {e}")

    async def send_startup_message(self):
        """봇 가동 시작 시 전송되는 메시지"""
        msg = "🚀 [V13.8.9] Sniper Master 엔진 가동 시작\n✅ 실시간 스캔 및 [Case 1,2,3] 저격을 시작합니다."
        await self.send_message(msg)

    async def get_updates(self):
        """텔레그램에서 사용자가 친 명령어(/status, /stop 등)를 읽어옵니다"""
        url = f"{self.base_url}/getUpdates"
        params = {"timeout": 5, "allowed_updates": ["message"]}
        if self.offset:
            params["offset"] = self.offset
        
        commands = []
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for result in data.get("result", []):
                        self.offset = result["update_id"] + 1
                        msg_data = result.get("message", {})
                        text = msg_data.get("text", "")
                        
                        if text.startswith("/"):
                            commands.append(text)
        except Exception:
            pass # 통신 지연 시 조용히 넘어감
            
        return commands

    async def handle_status(self, version):
        """ /status 명령어 처리 """
        msg = f"🟢 상태: 정상 가동 및 스캔 중\n⚙️ 버전: {version}"
        await self.send_message(msg)

    async def handle_balance(self):
        """ /balance 명령어 처리 """
        try:
            equity = await self.exchange.get_total_equity()
            msg = f"💰 현재 봇 운용 자산: {equity:.2f} USDT"
            await self.send_message(msg)
        except Exception as e:
            await self.send_message("❌ 잔고 조회에 실패했습니다.")

    async def close(self):
        """봇 종료 시 세션 닫기"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("🔒 텔레그램 세션 안전하게 종료됨.")