# 🚀 TradingBot AWS Operations Guide

본 문서는 AWS에서 V35 봇을 운영하고 업데이트하는 방법을 설명합니다.

## 1. 실시간 로그 확인 (Monitoring)
터미널에서 아래 명령어를 실행하여 봇의 상태를 실시간으로 확인할 수 있습니다.
```bash
ssh -i "C:\Users\zpwkg\Documents\WasherCRM\AWS_accesskey\WhiteOn-Key.pem" ubuntu@13.124.100.75 "tail -f /home/ubuntu/trading_bot/bot_live.log"
```

## 2. 지속적인 업데이트 (Continuous Update)
로컬 PC에서 코드를 수정한 후, 아래 명령어를 실행하면 AWS에 자동으로 반영되고 봇이 재시작됩니다.
```bash
python deploy_v35.py
```
*대상 파일: `v35_live.py`, `v30_train.py`, `config.py`, `exchange.py`, `telegram_bot.py`, `v29_env.py` 등*

## 3. 수동 제어 (Manual Control)
*   **봇 중지**: `ssh ... "pkill -f v35_live.py"`
*   **봇 시작**: `ssh ... "/home/ubuntu/trading_bot/start_v35_live.sh"`
*   **프로세스 확인**: `ssh ... "ps aux | grep v35_live.py"`

## 4. 서버 정보
*   **IP**: `13.124.100.75`
*   **User**: `ubuntu`
*   **Path**: `/home/ubuntu/trading_bot`
*   **Python**: `/home/ubuntu/trading_bot/venv/bin/python` (3.10)

---
**주의**: AWS 서버(t3.micro)의 리소스가 제한적이므로, 모델 훈련보다는 **실전 매매(Live Execution) 전용**으로 사용하는 것을 권장합니다.
