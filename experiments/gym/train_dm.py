import argparse
import os
import gin

from dm_control import suite
import torch.nn.functional as F
from torch import nn

import super_sac
from super_sac.wrappers import (
    DMControlWrapper,
    ParallelActorsDM,
)


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


class SharedEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self.fc0 = nn.Linear(dim, 128)
        self.fc1 = nn.Linear(128, dim)
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        x = F.relu(self.fc0(obs_dict["obs"]))
        x = F.relu(self.fc1(x))
        return x


def train_dm_control(args):
    gin.parse_config_file(args.config)

    def make_env():
        domain, task = args.env.split(".")
        env = suite.load(domain_name=domain, task_name=task)
        env = DMControlWrapper(env)
        return env

    train_env = ParallelActorsDM(make_env, args.parallel_envs)
    test_env = make_env()
    if args.render:
        train_env.reset()
        test_env.reset()

    # Extract observation dimension
    dim = train_env.get_observation_dim()

    act_space_size = train_env.action_spec.shape[-1]

    if args.shared_encoder:
        encoder = SharedEncoder(dim)
    else:
        encoder = IdentityEncoder(dim)

    # create agent
    agent = super_sac.Agent(
        act_space_size=act_space_size,
        encoder=encoder,
    )

    buffer = super_sac.replay.ReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method=args.logging,
        render=args.render,
        max_episode_steps=args.max_episode_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole.balance")
    parser.add_argument("--name", type=str, default="super_sac_cartpole_run")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    parser.add_argument("--shared_encoder", action="store_true")
    args = parser.parse_args()
    train_dm_control(args)
