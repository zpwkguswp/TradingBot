"""
V33-2 Stage 3 Hunter
====================
- Stage 3 고정 (단계 진화 없음)
- 총 10,000,000 스텝 탐색 (500,000 스텝씩 20회 반복)
- Rank #1 Score가 갱신될 때마다 자동 스냅샷 저장
- 모든 Rank 변동 기록을 hunt_log.csv에 누적
- 로그 파일 유지 (삭제/이동 없음) → 10만 스텝 단위 누적 분석 지속
"""

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
TOTAL_TARGET_STEPS = 10_000_000   # 총 탐색 스텝
STEPS_PER_RUN      = 5_000_000      # 회차당 스텝 (너무 작으면 콜백 오버헤드 큼)

LOG_DIR      = "v33_2_logs"
MODEL_DIR    = "elite_weights"
CONFIG_PATH  = "v33_2_config.json"
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.zip")
FINAL_MODEL  = os.path.join(MODEL_DIR, "v33_2_final_model.zip")
SNAPSHOT_DIR = os.path.join(MODEL_DIR, "stage3_snapshots")
HUNT_LOG     = os.path.join(LOG_DIR, "hunt_rank_log.csv")
TRADES_PAT   = os.path.join(LOG_DIR, "v33_2_trades_direct_*.csv")
TRADES_MAIN  = os.path.join(LOG_DIR, "v33_2_trades_main.csv")

STAGE3_CONFIG = {
    "stage": 3,
    "target_pf": 1.8,
    "learning_rate": 4e-6,
    "regret_penalty_weight": 6.0,
    "mae_penalty_threshold": -0.005,
    "capture_bonus_weight": 0.2,
    "total_timesteps": STEPS_PER_RUN,
    "description": f"Stage3 Hunter: {STEPS_PER_RUN:,} steps/run | 10M total"
}

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── 점수 계산 (v33_2_check_results.py 와 동일 공식) ───────────────────────────
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

# ── 로그에서 Rank 분석 ─────────────────────────────────────────────────────────
def analyze_ranks():
    """현재 로그 파일 전체를 읽어 10만 스텝 블록 기준 랭킹을 반환."""
    all_files = glob.glob(TRADES_PAT)
    if os.path.exists(TRADES_MAIN):
        all_files.append(TRADES_MAIN)

    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).drop_duplicates().sort_values("train_step")
    df['step_block'] = (df['train_step'] // 100_000) * 100_000

    blocks = []
    for step_val, group in df.groupby('step_block'):
        trades = len(group)
        if trades < 30:          # 최소 거래 건수 미달 블록 제외
            continue

        avg_pnl   = group['pnl'].mean() * 100
        total_pnl = group['pnl'].sum()  * 100
        wr        = (group['pnl'] > 0).mean() * 100

        pos_sum = group[group['pnl'] > 0]['pnl'].sum()
        neg_sum = abs(group[group['pnl'] < 0]['pnl'].sum())
        pf      = pos_sum / neg_sum if neg_sum > 0 else (pos_sum * 10.0 if pos_sum > 0 else 0.0)

        mdd = group['pnl'].expanding().sum().min() * -100
        mae = group.get('mae',    pd.Series([0]*trades)).mean() * 100
        reg = group.get('regret', pd.Series([0]*trades)).mean() * 100
        cap = group.get('capture',pd.Series([0]*trades)).mean() * 100

        score = calculate_score(avg_pnl, trades, pf, wr, mdd, mae, reg, cap / 100.0)

        blocks.append({
            "step_block": step_val,
            "Step"  : f"{step_val:,} ~ {step_val+100_000:,}",
            "Score" : score,
            "PnL"   : total_pnl,
            "Trades": trades,
            "WR"    : wr,
            "PF"    : pf,
            "MDD"   : mdd,
            "MAE"   : mae,
            "Regret": reg,
            "Capture": cap,
        })

    if not blocks:
        return pd.DataFrame()

    return pd.DataFrame(blocks).sort_values("Score", ascending=False).reset_index(drop=True)

# ── Rank #1 스냅샷 저장 ────────────────────────────────────────────────────────
def save_snapshot(run_idx, rank1_row, cumulative_steps):
    """best_model.zip → snapshot 폴더에 복사 + 메타정보 저장."""
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    score_str = f"{rank1_row['Score']:.2f}".replace("-", "neg")
    pf_str    = f"{rank1_row['PF']:.2f}"
    snap_name = f"run{run_idx:03d}_score{score_str}_pf{pf_str}_{ts}"
    snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
    os.makedirs(snap_path, exist_ok=True)

    # 모델 저장 (best_model 우선, 없으면 final_model)
    src = BEST_MODEL if os.path.exists(BEST_MODEL) else FINAL_MODEL
    if os.path.exists(src):
        shutil.copy(src, os.path.join(snap_path, "model.zip"))

    # 메타 JSON 저장
    meta = {
        "run"             : int(run_idx),
        "cumulative_steps": int(cumulative_steps),
        "timestamp"       : ts,
        "rank1_step"      : str(rank1_row.get("Step", "")),
        "score"           : float(rank1_row["Score"]),
        "pf"              : float(rank1_row["PF"]),
        "pnl"             : float(rank1_row["PnL"]),
        "trades"          : int(rank1_row["Trades"]),
        "wr"              : float(rank1_row["WR"]),
        "mdd"             : float(rank1_row["MDD"]),
        "mae"             : float(rank1_row["MAE"]),
        "regret"          : float(rank1_row.get("Regret", 0.0)),
        "capture"         : float(rank1_row.get("Capture", 0.0)),
    }
    with open(os.path.join(snap_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n  [★ SNAPSHOT] {snap_name}")
    print(f"      Score={rank1_row['Score']:.2f} | PF={rank1_row['PF']:.2f} | PnL={rank1_row['PnL']:.2f}% | WR={rank1_row['WR']:.1f}%")
    return snap_name

# ── Hunt 로그 기록 ─────────────────────────────────────────────────────────────
def log_rank_update(run_idx, cumulative_steps, rank_df, snapshot_name=""):
    """Rank가 바뀔 때마다 hunt_rank_log.csv에 기록."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i, row in rank_df.iterrows():
        rows.append({
            "logged_at"       : now,
            "run"             : run_idx,
            "cumulative_steps": cumulative_steps,
            "rank"            : i + 1,
            "step_block"      : row.get("Step", ""),
            "score"           : round(row["Score"], 2),
            "pf"              : round(row["PF"], 2),
            "pnl"             : round(row["PnL"], 2),
            "trades"          : row["Trades"],
            "wr"              : round(row["WR"], 1),
            "mdd"             : round(row["MDD"], 1),
            "mae"             : round(row["MAE"], 2),
            "snapshot"        : snapshot_name if i == 0 else "",
        })
    pd.DataFrame(rows).to_csv(
        HUNT_LOG, mode='a',
        header=not os.path.exists(HUNT_LOG),
        index=False
    )

# ── 랭킹 출력 ──────────────────────────────────────────────────────────────────
def print_ranks(rank_df, run_idx, cumulative_steps):
    w = 140
    print("=" * w)
    print(f"  [Stage3 Hunter] Run {run_idx} | 누적 스텝: {cumulative_steps:,} / {TOTAL_TARGET_STEPS:,}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * w)
    print(f"{'Rank':<5} {'Step Interval':<26} {'Score':>8} │ {'Tot_PnL':>10} {'Trades':>7} {'WR':>7} {'PF':>7} │ {'MDD':>7} {'MAE':>7} {'Reg':>7} {'Cap':>7}")
    print("-" * w)
    for i, row in rank_df.iterrows():
        star = "★" if i == 0 else "  "
        print(
            f"{i+1:<2} {star} {row['Step']:<25} {row['Score']:>8.1f} │"
            f" {row['PnL']:>9.2f}% {row['Trades']:>7} {row['WR']:>6.1f}% {row['PF']:>7.2f} │"
            f" {row['MDD']:>6.1f}% {row['MAE']:>7.2f} {row['Regret']:>7.2f} {row['Capture']:>6.1f}%"
        )
    print("=" * w)

# ── 메인 루프 ──────────────────────────────────────────────────────────────────
def main():
    # 1. Config를 Stage 3로 고정
    with open(CONFIG_PATH, "w") as f:
        json.dump(STAGE3_CONFIG, f, indent=4)
    print(f"\n[*] Config 설정 완료: Stage 3 고정")
    print(f"[*] 회차당 스텝: {STEPS_PER_RUN:,} | 총 목표: {TOTAL_TARGET_STEPS:,}")
    print(f"[*] 스냅샷 저장 경로: {SNAPSHOT_DIR}\n")

    total_runs      = TOTAL_TARGET_STEPS // STEPS_PER_RUN   # = 20
    cumulative_steps = 0
    best_rank1_score = -np.inf  # 지금까지 Rank #1의 최고 Score
    best_snapshot    = ""

    for run_idx in range(1, total_runs + 1):
        cumulative_steps += STEPS_PER_RUN

        print(f"\n{'='*60}")
        print(f" [Stage3 Hunter] Run {run_idx}/{total_runs} 시작")
        print(f" 누적 목표: {cumulative_steps:,} / {TOTAL_TARGET_STEPS:,} 스텝")
        print(f"{'='*60}\n")

        # 2. 학습 실행
        result = subprocess.run(
            [sys.executable, "v33_2_transfer_learning.py"],
            check=False  # 학습 중 오류가 나도 supervisor는 계속 진행
        )
        if result.returncode != 0:
            print(f"[!] 학습 프로세스 비정상 종료 (returncode={result.returncode}). 다음 회차 진행...")

        # 3. Rank 분석
        rank_df = analyze_ranks()
        if rank_df.empty:
            print("[?] 아직 충분한 거래 로그 없음. 다음 회차 진행...")
            continue

        # 4. Rank #1 확인
        rank1 = rank_df.iloc[0]
        current_score = rank1["Score"]

        print_ranks(rank_df, run_idx, cumulative_steps)

        snapshot_name = ""
        if current_score > best_rank1_score:
            print(f"\n[!!!] Rank #1 갱신! {best_rank1_score:.2f} → {current_score:.2f}")
            best_rank1_score = current_score
            snapshot_name    = save_snapshot(run_idx, rank1, cumulative_steps)
            best_snapshot    = snapshot_name
        else:
            print(f"\n[─] Rank #1 미갱신 (현재 최고: {best_rank1_score:.2f})")

        # 5. Hunt 로그 기록
        log_rank_update(run_idx, cumulative_steps, rank_df, snapshot_name)

    # ── 최종 결과 출력 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" [Stage3 Hunter] 탐색 완료!")
    print(f" 총 스텝: {TOTAL_TARGET_STEPS:,}")
    print(f" 최고 Rank #1 Score: {best_rank1_score:.2f}")
    print(f" 최고 스냅샷: {best_snapshot}")
    print(f" 스냅샷 폴더: {SNAPSHOT_DIR}")
    print(f" 기록 로그: {HUNT_LOG}")
    print("=" * 60)

if __name__ == "__main__":
    main()
