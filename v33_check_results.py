import os
import time
import pandas as pd
import numpy as np

def get_file_info(path, label):
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        mtime = os.path.getmtime(path)
        elapsed = time.time() - mtime
        
        if elapsed < 60:
            time_str = f"{int(elapsed)}초 전"
        elif elapsed < 3600:
            time_str = f"{int(elapsed // 60)}분 {int(elapsed % 60)}초 전"
        else:
            time_str = f"{int(elapsed // 3600)}시간 {int((elapsed % 3600) // 60)}분 전"
            
        return f"  ✅ {label} : {size:.1f} MB  (최종 저장: {time_str})"
    return f"  ❌ {label} : 없음"

def main():
    print("============================================================")
    print(f"  [V33 훈련 상황판]  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("============================================================")

    print("\n  ─── 모델 파일 상태 ─────────────────────────────────")
    print(get_file_info('elite_weights/best_model.zip', 'EvalCallback Best'))
    print(get_file_info('elite_weights/v33_best_model.zip', 'V33 Best (복사본)'))
    print(get_file_info('elite_weights/v33_final_model.zip', 'V33 Final Model '))

    print("\n  ─── EvalCallback 평가 기록 ─────────────────────────")
    eval_log_path = "v33_logs/evaluations.npz"
    if os.path.exists(eval_log_path):
        try:
            data = np.load(eval_log_path)
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)
            n_episodes = results.shape[1]
            
            print(f"  총 평가 횟수: {len(timesteps)}회\n")
            print("     #          스텝       평균 보상(Score)    최소        최대    에피소드")
            print("  ────────────────────────────────────────────────────────────────────────")
            for i in range(len(timesteps)):
                star = " ★" if mean_rewards[i] == np.max(mean_rewards) else ""
                print(f"    {i+1:<2d}    {timesteps[i]:10,d}    {mean_rewards[i]:+13.4f}  {np.min(results[i]):+10.4f}  {np.max(results[i]):+10.4f}      {n_episodes}")
            print("  ────────────────────────────────────────────────────────────────────────")
            
            best_idx = np.argmax(mean_rewards)
            print(f"  🏆 최우수 Score: {mean_rewards[best_idx]:+8.4f} (at {timesteps[best_idx]:,d} steps)")
        except Exception as e:
            print(f"  ⏳ 로그 분석 중... ({e})")
    else:
        print("  ⏳ 평가 기록 대기 중 (100,000 스텝 마다 평가 진행)")

    print("\n  ─── 종합 훈련 성과 (Training Trades) ────────────────")
    trade_log_path = "v33_logs/v33_trades.csv"
    if os.path.exists(trade_log_path):
        try:
            df_trades = pd.read_csv(trade_log_path)
            if not df_trades.empty:
                total_trades = len(df_trades)
                win_rate = (df_trades['pnl'] > 0).mean() * 100
                total_pnl = df_trades['pnl'].sum() * 100 # 누적 수익률(%)
                
                # MDD 및 Peak Equity
                peak_equity = df_trades['equity'].max()
                max_mdd = df_trades['mdd'].max()
                
                # PnL / MDD (Risk-Adjusted Return)
                # MDD가 0일 경우를 대비해 아주 작은 값 추가
                pnl_mdd_ratio = total_pnl / (max_mdd * 100 + 1e-9) if max_mdd > 0 else total_pnl

                print(f"  📊 총 매매횟수   : {total_trades:,} 회")
                print(f"  📈 누적 수익률   : {total_pnl:+.2f} %")
                print(f"  📉 최대 낙폭(MDD): {max_mdd*100:.2f} %")
                print(f"  ⚖️  PnL / MDD     : {pnl_mdd_ratio:.4f}")
                print(f"  🏆 최고 자산(Peak): {peak_equity:.4f} ({peak_equity*100-100:+.2f}%)")
                print(f"  🎯 승률(Win Rate): {win_rate:.2f} %")
                
                print("\n  [최근 거래 히스토리]")
                tail_df = df_trades[['step', 'coin', 'pnl', 'equity', 'mdd']].tail(5)
                print("    스텝         코인ID      수익률      자산        MDD")
                print("  ────────────────────────────────────────────────────────")
                for _, row in tail_df.iterrows():
                    print(f"    {int(row['step']):<10,d}  {int(row['coin']):<3d}     {row['pnl']:+8.4f}    {row['equity']:.4f}    {row['mdd']*100:5.2f}%")
            else:
                print("  ⏳ 매매 데이터가 아직 충분하지 않습니다.")
        except Exception as e:
            print(f"  ⏳ 로그 분석 중... ({e})")
    else:
        print("  ⏳ 거래 로그 대기 중 (v33_trades.csv)")
    print("\n============================================================")

if __name__ == "__main__":
    main()
