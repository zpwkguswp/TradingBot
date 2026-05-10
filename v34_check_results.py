import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ── V34 설정 ──────────────────────────────────
LOG_DIR = "v34_logs"
MODEL_DIR = "v34_snapshots"
TRADES_MAIN = os.path.join(LOG_DIR, "v34_trades_main.csv")
HUNT_LOG = os.path.join(LOG_DIR, "v34_hunt_log.csv")

def calculate_score(pnl, trade_count, pf, wr, mdd, mae, regret, capture):
    # V33-3 정통 공식 100% 동일
    s_pnl = pnl * 10.0 if pnl > 0 else pnl * 30.0
    s_trade = min(trade_count * 0.1, 10.0)
    s_pf = (pf - 1.5) * 15.0
    s_wr = (wr - 55.0) * 1.0
    p_mdd = mdd * -0.2
    p_mae = mae * -0.5
    p_reg = regret * -0.3
    b_cap = capture * 20.0
    return s_pnl + s_trade + s_pf + s_wr + p_mdd + p_mae + p_reg + b_cap

def check_results():
    print("\n" + "="*160)
    print(f"  [V34 Grand Finale 실시간 랭킹 모니터링]  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*160)

    # 1. 헌트 히스토리 (V33-3 포맷)
    if not os.path.exists(HUNT_LOG):
        print("[-] 아직 훈련 기록(Hunt Log)이 없습니다. 10만 스텝 이후에 첫 정산이 시작됩니다.")
        return

    h_df = pd.read_csv(HUNT_LOG)
    if h_df.empty:
        print("[-] 데이터가 아직 비어있습니다.")
        return

    # 컬럼명 매핑 (v33-3와 동일하게 출력)
    h_df.columns = ["Time", "Step", "Score", "PnL", "Trades", "WR", "PF", "MDD", "MAE", "Reg", "Cap", "Snap"]
    
    # 스코어 기준 랭킹 정렬 및 Top 10 제한
    res_df = h_df.sort_values("Score", ascending=False).head(10).reset_index(drop=True)

    print(f"{'Rank':<5} {'Step Interval':<26} {'Score':>8} │ {'PnL':>9} {'Trades':>7} {'WR':>7} {'PF':>7} │ {'MDD':>7} {'MAE':>7} {'Reg':>7} {'Cap':>7} │ {'Status'}")
    print("-" * 175)

    for i, row in res_df.iterrows():
        star = "★" if i == 0 else "  "
        step_val = int(row['Step'])
        step_str = f"{step_val-100000:,} ~ {step_val:,}"
        status = f"[{row['Snap']}]" if str(row['Snap']).strip() == "SAVED" else ""
        
        print(f"{i+1:<2} {star} {step_str:<25} {row['Score']:>8.1f} │ {row['PnL']:>8.2f}% {int(row['Trades']):>7} {row['WR']:>6.1f}% {row['PF']:>7.2f} │ {row['MDD']:>6.1f}% {row['MAE']:>7.2f} {row['Reg']:>7.2f} {row['Cap']:>6.1f}% │ {status}")

    # 2. 전체 누적 성적 요약
    if os.path.exists(TRADES_MAIN):
        df_main = pd.read_csv(TRADES_MAIN)
        if not df_main.empty:
            total_trades = len(df_main)
            total_pnl = df_main['pnl'].sum() * 100
            win_rate = (df_main['pnl'] > 0).mean() * 100
            pos_sum = df_main[df_main['pnl'] > 0]['pnl'].sum()
            neg_sum = abs(df_main[df_main['pnl'] < 0]['pnl'].sum())
            overall_pf = pos_sum / neg_sum if neg_sum > 0 else 0
            
            print("\n" + "="*160)
            print(f" [V34 전체 누적 퍼포먼스 요약]")
            print("-" * 160)
            print(f" ▶ 총 거래 수: {total_trades}회 | 누적 수익률: {total_pnl:.2f}% | 평균 승률: {win_rate:.1f}% | 전체 PF: {overall_pf:.2f}")
            print("="*160)

if __name__ == "__main__":
    check_results()
