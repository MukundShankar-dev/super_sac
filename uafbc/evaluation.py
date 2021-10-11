import gym
import numpy as np
import torch


def run_env(agent, env, episodes, max_steps, render=False, verbosity=1, num_envs=1):
    episode_return_history = []
    if render:
        env.render("rgb_array")
    for episode in range(episodes):
        episode_return = 0.0
        state = env.reset()
        still_counts = np.expand_dims(np.array([1.0 for _ in range(num_envs)]), 1)
        for _ in range(max_steps):
            if still_counts.sum() == 0:
                break
            action = agent.forward(state, num_envs=num_envs)
            state, reward, done, _ = env.step(action)
            if render:
                env.render("rgb_array")
            episode_return += still_counts * reward
            still_counts *= 1.0 - done
        if verbosity:
            print(f"Episode {episode}:: {episode_return.mean().item()}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)


def exploration_noise(action, random_process):
    return np.clip(action + random_process.sample(), -1.0, 1.0)


def evaluate_agent(
    agent,
    env,
    eval_episodes,
    max_episode_steps,
    render=False,
    verbosity=0,
    num_envs=1,
):
    agent.eval()
    returns = run_env(
        agent,
        env,
        eval_episodes,
        max_episode_steps,
        render,
        verbosity=verbosity,
        num_envs=num_envs,
    )
    agent.train()
    mean_return = returns.mean()
    return mean_return
