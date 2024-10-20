import numpy as np
from dm_env import specs


class DMControlWrapper:
    def __init__(self, env):
        self._env = env
        self._action_spec = self._env.action_spec()
        self._observation_spec = self._env.observation_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._extract_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._extract_observation(time_step)
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        return obs, reward, done, {}

    def _extract_observation(self, time_step):
        obs = time_step.observation
        if isinstance(obs, dict):
            obs = np.concatenate([v for v in obs.values()], axis=-1)
        return {"obs": obs}

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def observation_spec(self):
        return self._observation_spec

    @property
    def action_space(self):
        return self._action_spec

    @property
    def observation_space(self):
        return self._observation_spec


class ParallelActorsDM:
    def __init__(self, make_env_fn, num_envs):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(infos)

    @property
    def action_spec(self):
        return self.envs[0].action_spec

    @property
    def observation_spec(self):
        return self.envs[0].observation_spec
