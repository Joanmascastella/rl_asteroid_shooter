# run.py

import glob, os
from stable_baselines3 import PPO
from asteroid_shooter_env import AsteroidShooterEnv

def find_latest_model(pattern="ppo_asteroids*.zip"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No saved model found matching " + pattern)
    return max(files, key=os.path.getmtime)

def main():
    # 1) Find and load the latest model
    path = find_latest_model()
    model = PPO.load(path)
    print(f"Loaded model from: {path}")

    # 2) Create a raw (non-vectorized) game environment
    env = AsteroidShooterEnv()

    # 3) Reset and get the first observation
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    # 4) Roll out one episode
    while not done:
        # get an action (scalar int)
        action, _ = model.predict(obs, deterministic=False)

        # step returns: obs, reward, done, truncated, info
        # note: your env uses (obs, reward, done, False, info)
        obs, reward, done, _, info = env.step(int(action))

        # reward is already a float
        total_reward += reward

        # render to your pygame window
        env.render()

    print(f"Episode finished, total reward = {total_reward:.2f}")

if __name__ == "__main__":
    main()
