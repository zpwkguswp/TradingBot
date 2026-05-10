import os
import json
import subprocess
import pandas as pd
import time
import shutil
import sys
import glob
from datetime import datetime

CONFIG_PATH = "v33_2_config.json"
LOG_DIR = "v33_2_logs"
TRADES_FILE = os.path.join(LOG_DIR, "v33_2_trades_main.csv")
HISTORY_DIR = os.path.join(LOG_DIR, "history")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def analyze_results(target_pf):
    all_files = glob.glob(os.path.join(LOG_DIR, "v33_2_trades_direct_*.csv"))
    if not all_files:
        print("[Supervisor] No rank-based trade logs found.")
        return False, 0.0, 0.0

    try:
        df_list = []
        for f in all_files:
            try:
                df_list.append(pd.read_csv(f))
            except: pass
        
        if not df_list: return False, 0.0, 0.0
        df = pd.concat(df_list, ignore_index=True)
        
        if len(df) < 50: # 최소 매매 건수 미달 시 평가 유예
            print(f"[Supervisor] Not enough trades yet ({len(df)}/50)")
            return False, 0.0, 0.0

        # PF 계산
        pos_sum = df[df['pnl'] > 0]['pnl'].sum()
        neg_sum = abs(df[df['pnl'] < 0]['pnl'].sum())
        pf = pos_sum / neg_sum if neg_sum > 0 else 10.0
        
        # MAE 평균
        mae = df['mae'].mean()
        
        print(f"\n[Supervisor] Analysis Result - PF: {pf:.2f}, MAE: {mae:.4f} (Target PF: {target_pf})")
        
        if pf >= target_pf:
            return True, pf, mae
        return False, pf, mae
    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        return False, 0.0, 0.0

def backup_logs(stage, iteration):
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(HISTORY_DIR, f"stage_{stage}_iter_{iteration}_{timestamp}")
    os.makedirs(backup_path)
    
    for f in os.listdir(LOG_DIR):
        if f.endswith(".csv") or f.endswith(".json") or f.endswith(".npz"):
            shutil.move(os.path.join(LOG_DIR, f), os.path.join(backup_path, f))
    print(f"[*] Logs backed up to {backup_path}")

def main():
    iteration = 1
    while True:
        config = load_config()
        stage = config['stage']
        target_pf = config['target_pf']
        
        print(f"\n" + "="*60)
        print(f" [V33-2 Auto-Evolution] Stage {stage} | Iteration {iteration}")
        print(f" Target PF: {target_pf} | Regret Weight: {config['regret_penalty_weight']}")
        print("="*60 + "\n")
        
        # 1. 학습 실행 (현재 가상환경의 파이썬 사용)
        subprocess.run([sys.executable, "v33_2_transfer_learning.py"], check=True)
        
        # 2. 결과 분석
        success, current_pf, current_mae = analyze_results(target_pf)
        
        # 3. 로그 백업 및 정리 (매 회차마다 깨끗하게 시작)
        backup_logs(stage, iteration)
        
        # 4. 진화 결정
        if success:
            print(f"\n[!!!] Goal Achieved! Evolving to next level...")
            config['stage'] += 1
            
            # [Add] High-Water Mark Logic: 실제 달성한 PF가 다음 목표보다 높으면 그 수치를 baseline으로 고정
            next_target = round(config['target_pf'] + 0.3, 1)
            config['target_pf'] = max(next_target, round(current_pf, 2))
            
            config['regret_penalty_weight'] = round(config['regret_penalty_weight'] + 2.0, 1) # 페널티 강화
            
            # [Add] Learning Rate Decay: 기존의 좋은 습관을 보존하기 위해 학습률을 20% 감쇄
            old_lr = config.get('learning_rate', 5e-6)
            config['learning_rate'] = float(f"{old_lr * 0.8:.8f}")
            
            config['description'] = f"Stage {config['stage']}: Target PF {config['target_pf']} | LR {config['learning_rate']}"
            save_config(config)
            print(f"[*] Evolution: PF -> {config['target_pf']}, LR -> {config['learning_rate']}, Regret -> {config['regret_penalty_weight']}")
            iteration = 1 # 스테이지 바뀌면 회차 초기화
        else:
            print(f"\n[?] Goal not met. Retrying with same settings...")
            iteration += 1
            
        time.sleep(5) # 잠시 휴식

if __name__ == "__main__":
    main()
