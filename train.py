# train.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from asteroid_shooter_env import AsteroidShooterEnv   

class RenderCallback(BaseCallback):
    """ Renders the first env in the VecEnv each step. """
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        env.render()
        return True

class RewardCallback(BaseCallback):
    """ Prints episodic reward and running mean when an episode ends. """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # `infos` is a list of info dicts, one per sub‐env
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                r = ep["r"]    # episode return
                self.episode_rewards.append(r)
                mean_r = np.mean(self.episode_rewards)
                print(f"Episode {len(self.episode_rewards)} → reward={r:.2f}, mean={mean_r:.2f}")
        return True

def main():
    # 1) Vectorized env with Monitor to collect 'episode' info
    env = DummyVecEnv([
        lambda: Monitor(AsteroidShooterEnv())
    ])

    # 2) PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        verbose=1,
        learning_rate=1e-3,
        batch_size=64,
        n_steps=512,
        clip_range=0.1,
        gae_lambda=0.8,
        ent_coef=0.1,
        tensorboard_log="./ppo_tensorboard/",
    )

    # # # 3) Train with both Render and Reward callbacks
    callbacks = CallbackList([RenderCallback(), RewardCallback()])
    model.learn(
        total_timesteps=1_000_000,
        callback=callbacks,
        tb_log_name="run_1"
    )

    # 4) Save
    model.save("ppo_asteroids")

    # 5) Watch a final rollout
    play_env = AsteroidShooterEnv()
    obs, _ = play_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, info = play_env.step(action)
        play_env.render()

if __name__ == "__main__":
    main()
