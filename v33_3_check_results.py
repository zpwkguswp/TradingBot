import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──────────────────────────────────
LOG_DIR = "v33_3_logs"
MODEL_DIR = "elite_weights"
TRADES_MAIN = os.path.join(LOG_DIR, "v33_2_trades_main.csv")
TRADES_PATTERN = os.path.join(LOG_DIR, "v33_2_trades_direct_*.csv")
HUNT_LOG = os.path.join(LOG_DIR, "v33_3_hunt_log.csv")

def calculate_score(pnl, trade_count, pf, wr, mdd, mae, regret, capture):
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
    # 1. 트레이드 로그 집계
    all_files = glob.glob(TRADES_PATTERN)
    if os.path.exists(TRADES_MAIN): all_files.append(TRADES_MAIN)
    
    if not all_files:
        print("[-] 아직 트레이드 로그가 없습니다.")
    else:
        dfs = []
        for f in all_files:
            try: dfs.append(pd.read_csv(f))
            except: pass
        
        if dfs:
            df = pd.concat(dfs).drop_duplicates().sort_values(["run", "train_step"]) if 'run' in dfs[0].columns else pd.concat(dfs).drop_duplicates().sort_values("train_step")
            
            # 과거 데이터 호환성 (run, stage 컬럼이 없는 경우)
            if 'run' not in df.columns: df['run'] = 1
            if 'stage' not in df.columns: df['stage'] = 1
            df['run'] = df['run'].fillna(1)
            df['stage'] = df['stage'].fillna(1)

            df['step_block'] = (df['train_step'] // 100000) * 100000
            
            blocks = []
            # Run, Stage, Step_Block 별로 그룹화
            for (run_val, stage_val, step_val), group in df.groupby(['run', 'stage', 'step_block']):
                trades = len(group)
                if trades == 0: continue
                avg_pnl = group['pnl'].mean() * 100
                total_pnl = group['pnl'].sum() * 100
                wr = (group['pnl'] > 0).mean() * 100
                pos_sum = group[group['pnl'] > 0]['pnl'].sum()
                neg_sum = abs(group[group['pnl'] < 0]['pnl'].sum())
                pf = pos_sum / neg_sum if neg_sum > 0 else (pos_sum * 10.0 if pos_sum > 0 else 0)
                mdd = group['pnl'].expanding().sum().min() * -100
                mae = group.get('mae', pd.Series([0]*trades)).mean() * 100
                reg = group.get('regret', pd.Series([0]*trades)).mean() * 100
                cap = group.get('capture', pd.Series([0]*trades)).mean() * 100
                score = calculate_score(avg_pnl, trades, pf, wr, mdd, mae, reg, cap/100.0)
                
                blocks.append({
                    "Run": int(run_val), "Stage": int(stage_val),
                    "Step": f"{step_val:,} ~ {step_val+100000:,}",
                    "Score": score, "PnL": total_pnl, "Trades": trades, "WR": wr, "PF": pf,
                    "MDD": mdd, "MAE": mae, "Reg": reg, "Cap": cap
                })
            
            res_df = pd.DataFrame(blocks).sort_values("Score", ascending=False).reset_index(drop=True)
            
            print("\n" + "="*160)
            print(f"  [V33-3 Curriculum 실시간 랭킹]  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*160)
            print(f"{'Rank':<5} {'Run':<4} {'Stg':<4} {'Step Interval':<26} {'Score':>8} │ {'PnL':>9} {'Trades':>7} {'WR':>7} {'PF':>7} │ {'MDD':>7} {'MAE':>7} {'Reg':>7} {'Cap':>7}")
            print("-" * 160)
            for i, row in res_df.iterrows():
                star = "★" if i == 0 else "  "
                print(f"{i+1:<2} {star} {row['Run']:<4} {row['Stage']:<4} {row['Step']:<25} {row['Score']:>8.1f} │ {row['PnL']:>8.2f}% {row['Trades']:>7} {row['WR']:>6.1f}% {row['PF']:>7.2f} │ {row['MDD']:>6.1f}% {row['MAE']:>7.2f} {row['Reg']:>7.2f} {row['Cap']:>6.1f}%")

    # 2. 헌트 히스토리 (전체 기록)
    if os.path.exists(HUNT_LOG):
        print("\n" + "="*175)
        print(f"  [V33-3 Hunt History (Run/Step Performance)]")
        print("="*175)
        h_df = pd.read_csv(HUNT_LOG)
        
        # 각 Run의 Rank #1 기록만 필터링 (히스토리 가독성)
        history = h_df[h_df['rank'] == 1].copy()
        
        if not history.empty:
            print(f"{'Time':<20} {'Run':<4} {'Stg':<4} {'Step Interval':<25} {'Score':>8} │ {'PnL':>9} {'Trades':>7} {'WR':>7} {'PF':>7} │ {'MDD':>7} {'MAE':>7} {'Reg':>7} {'Cap':>7} │ {'Status'}")
            print("-" * 200)
            for _, row in history.tail(20).iterrows():
                status = "[SAVED]" if str(row.get('snap', '')).strip() == "SAVED" else ""
                # 컬럼명 대소문자 대응
                reg = row.get('Regret') if 'Regret' in row else row.get('Reg', 0)
                cap = row.get('Capture') if 'Capture' in row else row.get('Cap', 0)
                # 글로벌 스텝 범위 계산
                cum_steps = row.get('cum_steps', 0)
                if cum_steps > 0:
                    global_step_range = f"{cum_steps-500000:,} ~ {cum_steps:,}"
                else:
                    global_step_range = row.get('Step', 'Unknown')
                
                print(f"{row['at']:<20} {row['run']:<4} {row['stage']:<4} {global_step_range:<25} {row['Score']:>8.2f} │ {row['PnL']:>8.2f}% {row['Trades']:>7} {row['WR']:>6.1f}% {row['PF']:>7.2f} │ {row['MDD']:>6.1f}% {row['MAE']:>7.2f} {reg:>7.2f} {cap:>6.1f}% │ {status}")
        else:
            print("[-] 아직 기록된 히스토리가 없습니다.")
    
    print("\n" + "="*150)

if __name__ == "__main__":
    check_results()
