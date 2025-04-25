import argparse
import gym
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from garage.utils.common import PROJECT_ROOT, ENV_ABBRV_TO_FULL

# Monkey-patch to add missing aliases for SAC policies
ActorCriticPolicy.actor = property(lambda self: self.action_net)
ActorCriticPolicy.critic = property(lambda self: self.value_net)
ActorCriticPolicy.critic_target = property(lambda self: self.value_net)


def evaluate(env_name: str, model_path: str, n_episodes: int = 10, max_steps: int = 1000, deterministic: bool = True, algo: str = "sac"):
    env = gym.make(env_name)
    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    returns = []
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = model.predict(obs, deterministic=deterministic)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        returns.append(total_reward)
        print(f"Episode {ep+1}: return={total_reward}")
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    print(f"Average return over {n_episodes} episodes: {mean_ret} ± {std_ret}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC expert model")
    parser.add_argument("--env", type=str.lower, choices=list(ENV_ABBRV_TO_FULL.keys()), required=True, help="Environment abbreviation to evaluate on")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="RL algorithm of the saved model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    args = parser.parse_args()

    env_full = ENV_ABBRV_TO_FULL[args.env]
    model_path = Path(PROJECT_ROOT) / "experts" / env_full / "best_model"
    evaluate(env_full, str(model_path), args.episodes, args.max_steps, args.deterministic, args.algo)
