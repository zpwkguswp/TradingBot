import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──────────────────────────────────
LOG_DIR = "v33_2_logs"
MODEL_DIR = "elite_weights"
TRADES_MAIN = os.path.join(LOG_DIR, "v33_2_trades_main.csv")
TRADES_PATTERN = os.path.join(LOG_DIR, "v33_2_trades_direct_*.csv")
BEST_MODEL = os.path.join(MODEL_DIR, "best_model.zip")

def calculate_score(pnl, trade_count, pf, wr, mdd, mae, regret, capture):
    # 가중치 설정 (V33-2 최적화)
    # PnL(25), Trades(10), PF(20), WR(15), MDD(10), MAE(5), Regret(5), Capture(10)
    
    # 1. PnL 점수 (평균 수익률 기반)
    s_pnl = pnl * 10.0 if pnl > 0 else pnl * 30.0
    
    # 2. 매매 빈도 점수 (건당 0.1점, 최대 10점)
    s_trade = min(trade_count * 0.1, 10.0)
    
    # 3. PF 점수 (1.5 기준)
    s_pf = (pf - 1.5) * 15.0
    
    # 4. 승률 점수 (55% 기준)
    s_wr = (wr - 55.0) * 1.0
    
    # 5. 리스크 및 정밀도 페널티/보너스
    p_mdd = mdd * -0.2
    p_mae = mae * -0.5
    p_reg = regret * -0.3
    b_cap = capture * 20.0 # 캡처율이 높을수록 가산점
    
    return s_pnl + s_trade + s_pf + s_wr + p_mdd + p_mae + p_reg + b_cap

def check_results():
    # 1. 파일 취합
    all_files = glob.glob(TRADES_PATTERN)
    if os.path.exists(TRADES_MAIN): all_files.append(TRADES_MAIN)
    
    if not all_files:
        print("[-] No trade logs found.")
        return

    dfs = []
    for f in all_files:
        try: dfs.append(pd.read_csv(f))
        except: pass
    
    if not dfs: return
    df = pd.concat(dfs).drop_duplicates().sort_values("train_step")
    
    # 2. 10만 스텝 단위 그룹화
    df['step_block'] = (df['train_step'] // 100000) * 100000
    
    blocks = []
    for step_val, group in df.groupby('step_block'):
        trades = len(group)
        if trades == 0: continue
        
        avg_pnl = group['pnl'].mean() * 100
        total_pnl = group['pnl'].sum() * 100
        wr = (group['pnl'] > 0).mean() * 100
        
        pos_sum = group[group['pnl'] > 0]['pnl'].sum()
        neg_sum = abs(group[group['pnl'] < 0]['pnl'].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else (pos_sum * 10.0 if pos_sum > 0 else 0)
        
        mdd = group['pnl'].expanding().sum().min() * -100
        mae = group.get('mae', pd.Series([0]*len(group))).mean() * 100
        reg = group.get('regret', pd.Series([0]*len(group))).mean() * 100
        cap = group.get('capture', pd.Series([0]*len(group))).mean() * 100
        disp = group.get('entry_disp', pd.Series([0]*len(group))).mean() * 100
        cnt_ratio = group.get('is_counter', pd.Series([0]*len(group))).mean() * 100
        
        score = calculate_score(avg_pnl, trades, pf, wr, mdd, mae, reg, cap/100.0)
        
        blocks.append({
            "Step": f"{step_val:,} ~ {step_val+100000:,}",
            "Score": score,
            "PnL": total_pnl,
            "Trades": trades,
            "WR": wr,
            "PF": pf,
            "MDD": mdd,
            "MAE": mae,
            "Regret": reg,
            "Capture": cap,
            "Disp": disp,
            "Count": cnt_ratio
        })
    
    result_df = pd.DataFrame(blocks).sort_values("Score", ascending=False).reset_index(drop=True)
    result_df.index += 1 # 1위부터 표시
    
    # 3. 출력
    print("="*160)
    print(f"  [V33_2 전이학습 10만 스텝 단위 성적 랭킹]  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*160)
    
    print(f"{'Rank':<5} {'Step Interval':<25} {'Score':<10} │ {'Tot_PnL':>10} {'Trades':>8} {'WR':>8} {'PF':>8} │ {'MDD':>8} {'MAE':>8} {'Reg':>8} {'Cap':>8} {'Disp':>7} {'Count':>7}")
    print("-" * 160)
    
    for idx, row in result_df.iterrows():
        star = "★" if idx == 1 else "  "
        print(f"{idx:<2} {star} {row['Step']:<25} {row['Score']:>8.1f} │ {row['PnL']:>9.2f}% {row['Trades']:>8} {row['WR']:>7.1f}% {row['PF']:>8.2f} │ {row['MDD']:>7.1f}% {row['MAE']:>7.2f} {row['Regret']:>8.2f} {row['Capture']:>7.1f}% {row['Disp']:>6.2f}% {row['Count']:>6.1f}%")

    print("-" * 120)
    
    # 전체 누적 요약
    total_wr = (df['pnl'] > 0).mean() * 100
    total_sum_pnl = df['pnl'].sum() * 100
    print(f"  ─── 누적 전체 요약 ───  매매: {len(df):,}건 | 승률: {total_wr:.2f}% | 누적 수익률: {total_sum_pnl:+.2f}%")
    print("="*160)

if __name__ == "__main__":
    check_results()
