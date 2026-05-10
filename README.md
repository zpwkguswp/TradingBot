# 🚀 Bybit PPO Universal Trading Bot

> [!IMPORTANT]
> **V33 전이학습 수행 전 필독**: [V33 디버깅 사후 보고서](C:\Users\zpwkg\.gemini\antigravity\knowledge\v33_debugging\artifacts\v33_debug_postmortem.md)를 먼저 확인하세요.
> h1_ema_200 필터 문제 및 Windows 멀티프로세싱 CSV 충돌 방지 로직이 포함되어 있습니다.

Bybit 거래소 기반의 강화학습(PPO) 퀀트 트레이딩 봇 프로젝트입니다. 2020년부터 현재까지의 빅데이터를 활용하여 멀티 코인(50종)에 대한 범용 알파(Universal Alpha) 모델을 훈련하고 실전 매매를 수행합니다.

## 📌 주요 버전 및 히스토리

*   **V29 (Live)**: `v29_bybit_live.py` - 현재 실전 매매를 담당하는 피닉스 엔진입니다. 동적 비중 조절 및 스왑 로직이 포함되어 있습니다.
*   **V30 (Curriculum)**: 단계별 커리큘럼 학습(BTC/ETH -> Top15 -> 50종)을 도입한 초기 대함대 훈련 엔진입니다.
*   **V31 (Data)**: `v31_full_history_scraper.py` - 2020년부터의 5분봉 풀 히스토리 데이터를 수집하는 엔진입니다.
*   **V32**: `v32_train_from_scratch.py` - 시계열 분할(Time-Series Split)을 적용하여 2026년 장세에 최적화된 모델을 백지상태에서 훈련하는 최신 엔진입니다.
*   **V33 (Transfer Learning)**: `v33_2_transfer_learning.py` - V30의 지식을 V33 환경(풀 히스토리)으로 전이학습하여 정밀화하는 최신 엔진입니다. MAE/MFE 기반 보상 체계가 적용되었습니다.

## 📂 주요 파일 구조

| 파일명 | 역할 |
| :--- | :--- |
| `v29_bybit_live.py` | 실전 라이브 트레이딩 실행 스크립트 |
| `v32_train_from_scratch.py` | 최신 V32 PPO 모델 훈련 스크립트 |
| `v32_check_results.py` | 훈련 진행 상황 및 EvalCallback 로그 모니터링 |
| `v29_env.py` | 강화학습용 커스텀 Gym 환경 (ATR 정규화 보상 적용) |
| `v30_train.py` | TCN6Layer 아키텍처 및 커리큘럼 학습 로직 정의 |
| `config.py.example` | API 키 및 전략 설정 샘플 (실제 사용 시 `config.py`로 복사) |

## 🛠️ 설치 및 시작하기

1.  **환경 구축**:
    ```bash
    # 가상환경 생성 (Python 3.11 권장)
    python -m venv venv311
    source venv311/Scripts/activate

    # 의존성 설치
    pip install -r requirements.txt
    ```

2.  **설정**:
    `config.py.example` 파일을 `config.py`로 복사한 후, Bybit API Key와 Telegram Token을 입력합니다.

3.  **훈련 시작 (V32)**:
    ```bash
    python v32_train_from_scratch.py
    ```

4.  **라이브 가동 (V29)**:
    ```bash
    python v29_bybit_live.py
    ```

## ⚠️ 주의사항 (Security)

*   `config.py`에는 민감한 API 정보가 포함되어 있으므로 절대로 Git에 커밋하지 마세요. (이미 `.gitignore`에 등록되어 있습니다.)
*   훈련 데이터(`.parquet`) 및 모델 가중치(`.zip`)는 로컬 `data_storage/` 및 `elite_weights/` 폴더에서 관리됩니다.

## 📈 모니터링

훈련 중에는 다음 명령어로 실시간 성능을 확인할 수 있습니다:
```bash
python v32_check_results.py --watch
```
