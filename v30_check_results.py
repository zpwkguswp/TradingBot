import os
import pandas as pd
import numpy as np

LOG_DIR = "v30_logs"

def check_v30_curriculum_results():
    if not os.path.exists(LOG_DIR):
        print(f"⚠️ 로그 폴더({LOG_DIR})가 존재하지 않습니다. 아직 훈련이 시작되지 않았습니다.")
        return

    print("\n" + "="*70)
    print(" 🚀 [V30 Universal Alpha Fleet - 전황 브리핑 (Curriculum Logs)] 🚀")
    print("="*70)

    summary_data = []

    # Stage 1 ~ 3 까지의 로그 파일 순회
    for stage in [1, 2, 3]:
        log_file = os.path.join(LOG_DIR, f"v30_stage{stage}_trades.csv")
        
        if not os.path.exists(log_file):
            continue

        try:
            df = pd.read_csv(log_file)
            if df.empty:
                continue

            # 전적 계산
            total_trades = len(df)
            wins = len(df[df['pnl'] > 0])
            losses = len(df[df['pnl'] < 0])
            
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
            
            gross_profit = df[df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
            net_pnl = gross_profit - gross_loss
            
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            summary_data.append({
                "Stage": f"Stage {stage}",
                "Trades": total_trades,
                "Win Rate": f"{win_rate:.2f}%",
                "Net PnL (%)": f"{net_pnl:.2f}%",
                "Profit Factor": f"{pf:.2f}",
                "Gross Profit": f"{gross_profit:.2f}%",
                "Gross Loss": f"-{gross_loss:.2f}%"
            })
            
        except Exception as e:
            print(f"⚠️ Stage {stage} 로그 분석 중 에러 발생: {e}")

    if not summary_data:
        print("⚠️ 아직 기록된 전투(Trade) 로그가 없습니다. 모델이 탐색 중입니다.")
        return

    # 데이터프레임으로 변환하여 깔끔하게 출력
    res_df = pd.DataFrame(summary_data)
    
    # 컬럼 정렬 유지
    res_df = res_df[["Stage", "Trades", "Win Rate", "Net PnL (%)", "Profit Factor", "Gross Profit", "Gross Loss"]]
    
    print(res_df.to_string(index=False))
    print("="*70)
    print(" 💡 [지휘관 가이드]")
    print(" - Win Rate (승률): 40% 이상이면 준수, 50% 이상이면 S급 타점")
    print(" - Profit Factor (수익비): 1.5 이상이면 우상향 보장, 2.0 이상이면 성배(Holy Grail)")
    print(" - Net PnL (누적 수익): 훈련이 진행될수록 이 값이 폭발적으로 증가해야 합니다.")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_v30_curriculum_results()
