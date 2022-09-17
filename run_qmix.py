import os
from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import stunning_robots
import argparse
import gym.spaces as spaces
import stunning_robots
import supersuit as ss
from ray.rllib.algorithms.qmix import QMixConfig
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from wrapper import PettingZooParallelToRlLibMultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.tune.logger import pretty_print

checkpoints_path = "checkpoints"


def build(ckpt: int = None, config_path: str = None):
    def env_creator():
        if config_path is None:
            _env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG, auto_render=True, max_steps=500)
        else:
            _env = stunning_robots.parallel_env(**stunning_robots.load_map_config(config_path), auto_render=True,
                                                max_steps=500)
        # _env = stunning_robots.aec_env(**stunning_robots.DEFAULT_CONFIG, max_steps=10000)
        return _env

    env_name = "stunning_robots"

    test_env = PettingZooParallelToRlLibMultiAgentEnv(env_creator())
    obs_space = spaces.Tuple([test_env.observation_space for _ in test_env.agent_ids])
    act_space = spaces.Tuple([test_env.action_space for _ in test_env.agent_ids])
    # print(act_space)
    grouping = {
        "group_0": list(test_env.agent_ids)
    }

    register_env(env_name, lambda cfg: PettingZooParallelToRlLibMultiAgentEnv(env_creator())
                 .with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))
    # ModelCatalog.register_custom_model("am_vision_model", TorchActionMaskModel)
    # wrap the pettingzoo env in MultiAgent RLLib

    config = QMixConfig().rollouts(
        rollout_fragment_length=50,
    ).training(
        mixer="vdn",
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            # Size of the replay buffer in batches
            "capacity": 1000,
            # Choosing `fragments` here makes it so that the buffer stores entire
            # batches, instead of sequences, episodes or timesteps.
            "storage_unit": "fragments",
            "learning_starts": 1000,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
        }
    ).evaluation(
        evaluation_config={
            "render_env": True,
            "explore": False
        },
        evaluation_num_workers=0,
        evaluation_interval=1,
        evaluation_duration=1,
    ).exploration(
        exploration_config={
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.5,
            "final_epsilon": 0.01,
            # Timesteps over which to anneal epsilon.
            "epsilon_timesteps": 400000,
        }
    )
    config.simple_optimizer = True

    print(config.to_dict())
    # Build an Algorithm object from the config and run 1 training iteration.
    qmix = config.build(env=env_name)

    if ckpt is not None:
        qmix.restore(os.path.join(checkpoints_path, f"qmix-{ckpt}", f"checkpoint_{ckpt:06d}"))

    return qmix


def train(qmix, cur_ckpt, episode=500):
    if cur_ckpt is None:
        cur_ckpt = 0

    for _ in range(1, episode + 1):
        result = qmix.train()
        print(pretty_print(result))

        if _ % 25 == 0:
            checkpoint = qmix.save(os.path.join(checkpoints_path, f"qmix-{cur_ckpt + _}"))
            print("checkpoint saved at", checkpoint)
    return qmix, cur_ckpt + episode


def test(qmix, config_path: str = None, record=True):
    if config_path is None:
        env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG, max_steps=500)
    else:
        env = stunning_robots.parallel_env(**stunning_robots.load_map_config(config_path), max_steps=500)
    env = PettingZooParallelToRlLibMultiAgentEnv(env)
    grouping = {"group_0": list(env.agent_ids)}
    obs_space = spaces.Tuple([env.observation_space for _ in env.agent_ids])
    act_space = spaces.Tuple([env.action_space for _ in env.agent_ids])
    env = env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
    observations, rewards, dones, infos = None, None, None, None
    observations = env.reset()

    frames: List[np.ndarray] = list()

    while not dones or not dones['__all__']:
        env.render()
        if record:
            frames.append(env.render('rgb_array'))
        observations = tuple(OrderedDict({'action_mask': obs['action_mask'],
                                          'obs': obs['obs'],
                                          'state': obs['state']})
                             for obs in observations['group_0'])
        actions = qmix.compute_action(observations)
        observations, rewards, dones, infos = env.step(actions)

    if record:
        if not os.path.exists("video"):
            os.makedirs("video")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # opencv3.0
        h, w, c = frames[0].shape
        video_writer = cv2.VideoWriter(os.path.join("video", "out.avi"), fourcc, 10, (w, h))

        for frame in frames:
            video_writer.write(np.flip(frame, 2))

        video_writer.release()
        print(f"video saved at {os.path.join('video', 'out.avi')}")

    return qmix


def find_checkpoints():
    dirs = os.listdir(checkpoints_path)
    ckpt = None
    for item in dirs:
        if item.startswith('qmix-'):
            ckpt = int(item[5:]) if ckpt is None else max(ckpt, int(item[5:]))
    return ckpt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train PPO to play in stunning-robots.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
    parser.add_argument("-r", "--record", action='store_true')
    args = parser.parse_args()
    checkpoints = find_checkpoints()
    qmix_algorithms = build(checkpoints)
    if args.command == "train":
        train(qmix_algorithms, checkpoints)
    elif args.command == "test":
        test(qmix_algorithms, record=args.record)
    else:
        parser.print_help()
