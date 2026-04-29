import optuna
import pandas as pd

STUDY_DB = "sqlite:///v29_overseer.db"
STUDY_NAME = "v29_universal_alpha_v1.0"

def check_results():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
        df = study.trials_dataframe()
        
        if df.empty:
            print("No trials completed yet.")
            return

        df = df[df['state'] == 'COMPLETE']
        if df.empty:
            print("No completed trials yet.")
            return

        # 🌟 칼럼 매핑 (콘솔 출력 가독성 극대화)
        cols = {
            'value': 'Score',
            'params_timeframe': 'TF',
            'user_attrs_pf': 'PF',
            'user_attrs_avg_ret': 'Ret',
            'user_attrs_mdd': 'MDD',
            'user_attrs_trades': 'Trades',
            'params_sl_atr_coef': 'SL_ATR',
            'params_far_th': 'Far_th'
        }
        
        available_cols = [c for c in cols.keys() if c in df.columns]
        res = df[available_cols].copy()
        res = res.rename(columns={c: cols[c] for c in available_cols})
        res = res.sort_values('Score', ascending=False)

        # 🌟 데이터 포맷팅 (소수점 정리 및 % 변환)
        if 'PF' in res.columns: res['PF'] = res['PF'].map('{:.2f}'.format)
        if 'Ret' in res.columns: res['Ret'] = (res['Ret'] * 100).map('{:.2f}%'.format)
        if 'MDD' in res.columns: res['MDD'] = (res['MDD'] * 100).map('{:.2f}%'.format)
        if 'Score' in res.columns: res['Score'] = res['Score'].map('{:.3f}'.format)
        for col in ['SL_ATR', 'Far_th']:
            if col in res.columns: res[col] = res[col].map('{:.2f}'.format)

        print("\n==================================================================")
        print("                 [V29 Universal Alpha Leaderboard]                ")
        print("==================================================================")
        print(res.head(15).to_string(index=False))
        print("==================================================================\n")

        print("=== Performance by Timeframe (Avg Score) ===")
        if 'Score' in res.columns:
            # 텍스트로 바뀐 Score를 잠시 float로 계산
            temp_df = df[available_cols].rename(columns={c: cols[c] for c in available_cols})
            avg_score_by_tf = temp_df.groupby('TF')['Score'].mean().sort_values(ascending=False)
            print(avg_score_by_tf.map('{:.3f}'.format))
        else:
            print("No Score data yet.")

        best_trial = study.best_trial
        print(f"\n[BEST COMMANDER] Trial {best_trial.number}")
        print(f"Value: {best_trial.value:.4f}")
        print(f"Parameters: {best_trial.params}")

    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    check_results()