from v29_train import make_env
try:
    print("Initializing env...")
    env_fn = make_env("BTCUSDT", "1h", "train", {})
    env = env_fn()
    obs = env.reset()
    print("Success! obs shape:", obs[0].shape if isinstance(obs, tuple) else obs.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
