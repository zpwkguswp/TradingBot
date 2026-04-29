import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

# v29_train.py에 있는 클래스와 설정들을 그대로 가져옵니다
from v29_train import TCN6LayerExtractor, TICKERS, N_STACK, DEVICE, DATA_DIR
from v29_env import V29_Universal_Env

# 👑 [Trial 59 황금 파라미터 이식]
BEST_PARAMS = {
    "lr": 3.630667348192714e-05,
    "ent_coef": 0.02857890487638162,
    "far_th": 2.2044178686597258,
    "sl_atr_coef": 3.8350847145336115
}

def make_env(ticker, split):
    coin_id = TICKERS.index(ticker)
    def _init():
        return V29_Universal_Env(
            data_dir=DATA_DIR,
            coin_files=[f"{ticker}_2h.parquet"],
            coin_id=coin_id,
            split_type=split,
            target_profit=0.008,
            far_th=BEST_PARAMS["far_th"],
            sl_atr_coef=BEST_PARAMS["sl_atr_coef"],
            trail_act=0.020
        )
    return _init

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    print("[V29] Trial 59 Golden Recipe Final Model Build Starting...")
    
    # 10개 코인 통합 훈련장 생성
    venv = SubprocVecEnv([make_env(t, "train") for t in TICKERS])
    venv = VecFrameStack(venv, n_stack=N_STACK)
    
    policy_kwargs = dict(
        features_extractor_class=TCN6LayerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[128, 128])
    )
    
    # 모델 생성 (황금 파라미터 적용)
    model = PPO("MlpPolicy", venv, verbose=1, policy_kwargs=policy_kwargs,
                learning_rate=BEST_PARAMS["lr"], ent_coef=BEST_PARAMS["ent_coef"], 
                device=DEVICE)
                
    # 150만 스텝 훈련 (약 15~30분 소요 예상)
    TOTAL_STEPS = 1_500_000 
    print(f"[Training] Starting {TOTAL_STEPS} steps. Model will be saved automatically upon completion...")
    
    model.learn(total_timesteps=TOTAL_STEPS)
    
    # 영구 보존용 파일 저장
    os.makedirs("elite_weights", exist_ok=True)
    model_path = "elite_weights/v29_best_model_2h.zip"
    model.save(model_path)
    print("\n[Finish] Cultivation Complete! Model saved at: {model_path}")
    
    venv.close()
