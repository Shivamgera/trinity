"""Verify SB3 PPO training works on CartPole."""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.utils.seed import set_global_seed


def test_sb3():
    set_global_seed(42)

    # Create vectorized environment
    env = make_vec_env("CartPole-v1", n_envs=2, seed=42)

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        seed=42,
    )

    # Train for 1000 steps
    model.learn(total_timesteps=1000)

    # Evaluate
    eval_env = gym.make("CartPole-v1")
    obs, info = eval_env.reset(seed=42)
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Evaluation reward: {total_reward}")
    print(f"SB3 PPO training: PASSED (reward={total_reward})")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    test_sb3()
