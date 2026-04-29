"""
V32 훈련 중간 체크 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
훈련이 실행 중인 동안 별도 터미널에서 실행하면
EvalCallback이 기록한 평가 로그와 현재 최우수 모델
상태를 실시간으로 확인할 수 있습니다.

사용법:
  python v32_check_results.py           # 1회 출력
  python v32_check_results.py --watch   # 30초마다 자동 갱신
"""

import argparse
import os
import time
import numpy as np

LOG_DIR      = "v32_logs"
ELITE_DIR    = "elite_weights"
BEST_MODEL   = os.path.join(ELITE_DIR, "best_model.zip")         # EvalCallback 저장 위치
FINAL_MODEL  = os.path.join(ELITE_DIR, "v32_best_model_2h.zip")  # 훈련 완료 후 복사 위치
EVAL_LOG     = os.path.join(LOG_DIR, "evaluations.npz")          # EvalCallback 평가 기록


def _fmt_size(path: str) -> str:
    try:
        mb = os.path.getsize(path) / 1_048_576
        return f"{mb:.1f} MB"
    except Exception:
        return "?"


def check_model_status():
    """저장된 모델 파일 상태 확인."""
    print("\n  ─── 모델 파일 상태 ─────────────────────────────────")

    for label, path in [
        ("EvalCallback Best ", BEST_MODEL),
        ("V32 Final (복사본)", FINAL_MODEL),
    ]:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            age   = time.time() - mtime
            mins  = int(age // 60)
            secs  = int(age % 60)
            print(f"  ✅ {label}: {_fmt_size(path)}  (최종 저장: {mins}분 {secs}초 전)")
        else:
            print(f"  ⏳ {label}: 아직 저장되지 않음")


def check_eval_log():
    """EvalCallback이 기록한 evaluations.npz 파싱 및 출력."""
    if not os.path.exists(EVAL_LOG):
        print("\n  ─── 평가 로그 ──────────────────────────────────────")
        print("  ⏳ 아직 평가 기록 없음 (첫 50,000 스텝 완료 전)")
        return

    try:
        data = np.load(EVAL_LOG)
        timesteps = data["timesteps"]          # shape: (n_evals,)
        results   = data["results"]            # shape: (n_evals, n_episodes)
        ep_lens   = data.get("ep_lengths", None)

        n_evals = len(timesteps)

        print("\n  ─── EvalCallback 평가 기록 ─────────────────────────")
        print(f"  총 평가 횟수: {n_evals}회")
        print()
        print(f"  {'#':>4}  {'스텝':>10}  {'평균 보상':>10}  {'최소':>8}  {'최대':>8}  {'에피소드':>6}")
        print("  " + "─" * 56)

        best_mean = -np.inf
        best_step = 0

        for i in range(n_evals):
            step   = int(timesteps[i])
            ep_rew = results[i]          # 각 평가 에피소드 보상 배열
            mean_r = float(np.mean(ep_rew))
            min_r  = float(np.min(ep_rew))
            max_r  = float(np.max(ep_rew))
            n_ep   = len(ep_rew)
            marker = " ★" if mean_r == np.max(np.mean(results, axis=1)) else ""

            print(f"  {i+1:>4}  {step:>10,}  {mean_r:>+10.4f}  "
                  f"{min_r:>+8.4f}  {max_r:>+8.4f}  {n_ep:>6}{marker}")

            if mean_r > best_mean:
                best_mean = mean_r
                best_step = step

        print("  " + "─" * 56)
        print(f"  🏆 최우수: step={best_step:,}  mean_reward={best_mean:+.4f}")

        # 추세 분석 (최근 5회 vs 이전 5회)
        if n_evals >= 10:
            recent = np.mean(results[-5:])
            prev   = np.mean(results[-10:-5])
            delta  = recent - prev
            trend  = "📈 개선 중" if delta > 0 else "📉 정체/하락"
            print(f"  추세 (최근 5 vs 이전 5): {delta:+.4f}  → {trend}")

        # 에피소드 길이 (있을 경우)
        if ep_lens is not None and len(ep_lens) > 0:
            avg_len = float(np.mean(ep_lens[-1]))
            print(f"  최근 평균 에피소드 길이: {avg_len:.0f} steps")

    except Exception as e:
        print(f"  [Error] 평가 로그 읽기 실패: {e}")


def check_trade_log():
    """v32_logs 폴더 내 CSV 거래 로그 요약 출력."""
    print("\n  ─── 거래 로그 파일 ─────────────────────────────────")
    if not os.path.exists(LOG_DIR):
        print("  ⏳ 로그 폴더 없음")
        return

    found = False
    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".csv"):
            continue
        path  = os.path.join(LOG_DIR, fname)
        mtime = os.path.getmtime(path)
        age   = int((time.time() - mtime) // 60)
        try:
            import pandas as pd
            df = pd.read_csv(path)
            rows = len(df)
            print(f"  📄 {fname}: {rows}행  (최종 수정: {age}분 전)")
        except Exception:
            print(f"  📄 {fname}: (읽기 실패)")
        found = True

    if not found:
        print("  ⏳ CSV 로그 없음")


def run_check():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 60)
    print(f"  [V32 훈련 상황판]  {now}")
    print("=" * 60)

    check_model_status()
    check_eval_log()
    check_trade_log()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V32 훈련 중간 체크")
    parser.add_argument(
        "--watch", action="store_true",
        help="30초마다 자동 갱신 (훈련 중 모니터링)"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="--watch 갱신 간격(초), 기본 30"
    )
    args = parser.parse_args()

    if args.watch:
        print(f"  👀 Watch 모드: {args.interval}초마다 자동 갱신  (Ctrl+C로 종료)")
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                run_check()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  [Watch 종료]")
    else:
        run_check()
