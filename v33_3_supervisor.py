import os
import json
import subprocess
import glob
import shutil
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────────────────────
TOTAL_TARGET_STEPS = 10_000_000   # 총 목표 스텝
STEPS_PER_RUN      = 500_000      # 랭킹 체크 및 스냅샷을 위한 회차 분할 (0.5M씩 20회)

LOG_DIR      = "v33_3_logs"
MODEL_DIR    = "elite_weights"
CONFIG_PATH  = "v33_2_config.json"
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.zip")
FINAL_MODEL  = os.path.join(MODEL_DIR, "v33_2_final_model.zip")
SNAPSHOT_DIR = os.path.join(MODEL_DIR, "v33_3_snapshots")
HUNT_LOG     = os.path.join(LOG_DIR, "v33_3_hunt_log.csv")
TRADES_PAT   = os.path.join(LOG_DIR, "v33_2_trades_direct_*.csv")
TRADES_MAIN  = os.path.join(LOG_DIR, "v33_2_trades_main.csv")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── 점수 계산 공식 (v33_2 와 동일) ───────────────────────────
def calculate_score(pnl, trade_count, pf, wr, mdd, mae, regret, capture):
    s_pnl  = pnl * 10.0   if pnl > 0 else pnl * 30.0
    s_trade = min(trade_count * 0.1, 10.0)
    s_pf   = (pf - 1.5) * 15.0
    s_wr   = (wr - 55.0) * 1.0
    p_mdd  = mdd * -0.2
    p_mae  = mae * -0.5
    p_reg  = regret * -0.3
    b_cap  = capture * 20.0
    return s_pnl + s_trade + s_pf + s_wr + p_mdd + p_mae + p_reg + b_cap

# ── 랭킹 분석 ─────────────────────────────────────────────────────────
def analyze_ranks():
    all_files = glob.glob(TRADES_PAT)
    if os.path.exists(TRADES_MAIN): all_files.append(TRADES_MAIN)
    if not all_files: return pd.DataFrame()

    dfs = []
    for f in all_files:
        try: dfs.append(pd.read_csv(f))
        except: pass
    if not dfs: return pd.DataFrame()

    df = pd.concat(dfs).drop_duplicates()
    if 'run' not in df.columns: df['run'] = 1
    if 'stage' not in df.columns: df['stage'] = 1
    df['run'] = df['run'].fillna(1)
    df['stage'] = df['stage'].fillna(1)
    
    df = df.sort_values(["run", "train_step"])
    df['step_block'] = (df['train_step'] // 100_000) * 100_000

    blocks = []
    for (run_val, stage_val, step_val), group in df.groupby(['run', 'stage', 'step_block']):
        trades = len(group)
        if trades < 30: continue
        
        avg_pnl = group['pnl'].mean() * 100
        total_pnl = group['pnl'].sum() * 100
        wr = (group['pnl'] > 0).mean() * 100
        pos_sum = group[group['pnl'] > 0]['pnl'].sum()
        neg_sum = abs(group[group['pnl'] < 0]['pnl'].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else (pos_sum * 10.0 if pos_sum > 0 else 0.0)
        mdd = group['pnl'].expanding().sum().min() * -100
        mae = group.get('mae', pd.Series([0]*trades)).mean() * 100
        reg = group.get('regret', pd.Series([0]*trades)).mean() * 100
        cap = group.get('capture',pd.Series([0]*trades)).mean() * 100

        blocks.append({
            "Step": f"{step_val:,} ~ {step_val+100_000:,}",
            "Score": calculate_score(avg_pnl, trades, pf, wr, mdd, mae, reg, cap/100.0),
            "PnL": total_pnl, "Trades": trades, "WR": wr, "PF": pf, "MDD": mdd, "MAE": mae, "Regret": reg, "Capture": cap
        })
    return pd.DataFrame(blocks).sort_values("Score", ascending=False).reset_index(drop=True) if blocks else pd.DataFrame()

# ── 스냅샷 및 로그 ────────────────────────────────────────────────────────
def save_snapshot(run_idx, stage, rank1, cumulative_steps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_name = f"stage{stage}_run{run_idx:03d}_score{rank1['Score']:.1f}_{ts}"
    snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
    os.makedirs(snap_path, exist_ok=True)
    src = BEST_MODEL if os.path.exists(BEST_MODEL) else FINAL_MODEL
    if os.path.exists(src): shutil.copy(src, os.path.join(snap_path, "model.zip"))
    with open(os.path.join(snap_path, "meta.json"), "w") as f:
        json.dump({**rank1.to_dict(), "stage": stage, "cumulative_steps": cumulative_steps, "timestamp": ts}, f, indent=4)
    print(f"  [★ NEW RECORD] {snap_name}")

def log_rank(run_idx, stage, cumulative_steps, rank_df, snapshot_name=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i, row in rank_df.head(5).iterrows():
        rows.append({"at": now, "run": run_idx, "stage": stage, "cum_steps": cumulative_steps, "rank": i+1, **row.to_dict(), "snap": snapshot_name if i==0 else ""})
    pd.DataFrame(rows).to_csv(HUNT_LOG, mode='a', header=not os.path.exists(HUNT_LOG), index=False)

# ── 메인 실행 ──────────────────────────────────────────────────────────────────
def main():
    print(f"\n[*] V33-3 Curriculum Hunter 시작 (Total: {TOTAL_TARGET_STEPS:,} steps)")
    total_runs = TOTAL_TARGET_STEPS // STEPS_PER_RUN
    best_score = -np.inf
    cumulative_steps = 0
    start_run = 1
    
    # [Resume Logic] 기존 로그가 있다면 마지막 Run과 스텝 확인
    if os.path.exists(HUNT_LOG):
        try:
            h_df = pd.read_csv(HUNT_LOG)
            if not h_df.empty:
                last_run = h_df['run'].max()
                last_cum_steps = h_df[h_df['run'] == last_run]['cum_steps'].iloc[0]
                start_run = int(last_run) + 1
                cumulative_steps = int(last_cum_steps)
                # 최고 점수 복구
                best_score = h_df['Score'].max()
                print(f"[*] 기존 기록 발견: Run {last_run} ({cumulative_steps:,} steps) 완료됨.")
                print(f"[*] 최고 점수 기록: {best_score:.2f}")
                print(f"[*] Run {start_run}부터 재개합니다.")
        except Exception as e:
            print(f"[!] Resume 실패, 처음부터 시작합니다: {e}")

    for run_idx in range(start_run, total_runs + 1):
        # 1. 스테이지 결정 (330만 단위)
        if cumulative_steps < 3_333_333: stage = 1
        elif cumulative_steps < 6_666_666: stage = 2
        else: stage = 3

        # 2. 컨피그 업데이트
        config = {
            "stage": stage, "run_idx": run_idx, "cum_start_step": cumulative_steps,
            "log_dir": LOG_DIR, "target_pf": 1.5 + (stage-1)*0.2, "learning_rate": 5e-6 * (0.8**(stage-1)),
            "regret_penalty_weight": 4.0 + (stage-1)*2.0, "mae_penalty_threshold": -0.005,
            "capture_bonus_weight": 0.2, "total_timesteps": STEPS_PER_RUN,
            "description": f"V33-3 Stage {stage} | Run {run_idx}/{total_runs}"
        }
        with open(CONFIG_PATH, "w") as f: json.dump(config, f, indent=4)

        print(f"\n[Run {run_idx}/{total_runs}] Stage {stage} | 목표 누적: {cumulative_steps + STEPS_PER_RUN:,} 스텝")
        subprocess.run([sys.executable, "v33_2_transfer_learning.py"], check=True)
        cumulative_steps += STEPS_PER_RUN

        # 3. 결과 분석 및 랭킹 기록
        rank_df = analyze_ranks()
        if not rank_df.empty:
            rank1 = rank_df.iloc[0]
            snap_name = ""
            if rank1["Score"] > best_score:
                best_score = rank1["Score"]
                save_snapshot(run_idx, stage, rank1, cumulative_steps)
                snap_name = "SAVED"
            log_rank(run_idx, stage, cumulative_steps, rank_df, snap_name)
            print(f"  Rank #1: {rank1['Step']} | Score: {rank1['Score']:.2f} | PF: {rank1['PF']:.2f}")

    print(f"\n[*] 훈련 스케줄 완료. 총 {cumulative_steps:,} 스텝 완주.")

if __name__ == "__main__":
    main()
