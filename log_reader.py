import pickle
import json
import os
import sys
import time
from datetime import datetime

# Windows 콘솔 인코딩 대응
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 파일 경로
LOG_FILE = "v29_experience_dataset.pkl"
STATE_FILE = "v29_live_state.json"

def format_ts(ts):
    if not ts: return "N/A"
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')
        return ts[5:16].replace('T', ' ')
    except:
        return str(ts)

def read_dashboard():
    print("=" * 60)
    print("🚀 V29 UNIVERSAL ALPHA - 통합 전투 상황판")
    print("=" * 60)

    # 1. 현재 교전 중인 포지션 (Live Positions)
    print("\n⚔️ [1단계: 현재 교전 중인 병사들 - Active Positions]")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            positions = state.get("positions", {})
            if not positions:
                print("   > 현재 교전 중인 포지션이 없습니다. (Ready for orders)")
            else:
                for ticker, pos in positions.items():
                    side = pos.get('side', 'N/A').upper()
                    entry_px = pos.get('entry_price', 0.0)
                    mfe = pos.get('mfe', 0.0) * 100.0
                    mae = pos.get('mae', 0.0) * 100.0
                    entry_time = format_ts(pos.get('entry_timestamp'))
                    
                    print(f"   [{ticker}] {side} | 진입: {entry_px:.4f} ({entry_time})")
                    print(f"   ㄴ 실시간 MFE: {mfe:+.2f}% | MAE: {mae:+.2f}%")
        except Exception as e:
            print(f"   [!] 상태 파일 읽기 오류: {e}")
    else:
        print("   [!] 상태 정보를 찾을 수 없습니다.")

    # 2. 전사하거나 승리한 기록 (Past Experience)
    print("\n📜 [2단계: 과거 전투 기록 - Experience Archive]")
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "rb") as f:
                logs = pickle.load(f)
            
            if not logs:
                print("   > 아직 기록된 전투가 없습니다.")
            else:
                # 총 수익률 계산
                total_pnl = sum(log.get('performance', {}).get('pnl_pct', 0.0) for log in logs if 'performance' in log)
                win_count = sum(1 for log in logs if log.get('performance', {}).get('pnl_pct', 0.0) > 0)
                win_rate = (win_count / len(logs)) * 100.0 if logs else 0
                
                print(f"   [통합 성과] 총 데이터: {len(logs)}건 | 누적 수익: {total_pnl:+.2f}% | 승률: {win_rate:.1f}%")
                print("-" * 50)
                
                for i, data in enumerate(reversed(logs[-5:])):
                    entry = data.get('entry', {})
                    exit = data.get('exit', {})
                    perf = data.get('performance', {})
                    ticker = data.get('ticker', 'N/A')
                    pnl = perf.get('pnl_pct', 0.0)
                    
                    pnl_tag = "[WIN]" if pnl > 0 else "[LOSS]"
                    print(f"   {pnl_tag} {ticker} ({format_ts(entry.get('timestamp'))} ~ {format_ts(exit.get('timestamp'))})")
                    print(f"   ㄴ 최종 수익: {pnl:+.2f}% | MFE: {perf.get('mfe_pct', 0.0):.2f}% | MAE: {perf.get('mae_pct', 0.0):.2f}%")
                    print(f"   ㄴ 사유: {exit.get('reason', 'N/A')}")
        except Exception as e:
            print(f"   [!] 로그 파일 읽기 오류: {e}")
    else:
        print("   [!] 과거 기록 파일을 찾을 수 없습니다.")
    
    print("\n" + "=" * 60)
    print(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    read_dashboard()

