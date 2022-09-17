import os
import ray
from typing import List

import cv2
import numpy as np
import stunning_robots
import argparse
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from wrapper import PettingZooParallelToRlLibMultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from model.action_mask_model import TorchActionMaskModel

checkpoints_path = 'checkpoints'


def build(ckpt: int = None, config_path: str = None, max_steps=500, parallel=False):
    env_name = "stunning_robots"

    def env_creator():
        if config_path is None:
            _env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG, auto_render=True, max_steps=max_steps)
        else:
            _env = stunning_robots.parallel_env(**stunning_robots.load_map_config(os.path.join(os.path.dirname(
                __file__), 'stunning_robots', 'maps', config_path)), auto_render=True, max_steps=max_steps)
        # _env = stunning_robots.aec_env(**stunning_robots.DEFAULT_CONFIG, max_steps=10000)
        return _env

    test_env = PettingZooParallelToRlLibMultiAgentEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    num_agents = len(test_env.agent_ids)
    # print(act_space)
    # grouping = {
    #     "group_0": list(test_env.agent_ids)
    # }

    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)

    register_env(env_name, lambda cfg: PettingZooParallelToRlLibMultiAgentEnv(env_creator())
                 # .with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
                 )

    # ModelCatalog.register_custom_model("am_vision_model", TorchActionMaskModel)
    # wrap the pettingzoo env in MultiAgent RLLib
    def gen_policy(_):
        return (None, obs_space, act_space,
                {"model": {
                    "custom_model": "action_mask_model",
                },
                    "gamma": 0.99,
                })
    if parallel:
        policies = {f"{_}": gen_policy(_) for _ in range(num_agents)}
        policy_mapping = dict(zip(sorted(test_env.agent_ids), sorted(policies.keys())))
    else:
        policies = {"policy_0": gen_policy(0)}  # "policy_0"
        policy_mapping = {agent: "policy_0" for agent in test_env.agent_ids}  # "policy_0"
    print(policy_mapping)

    config = PPOConfig().framework(framework="torch").training(
        use_gae=True,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        train_batch_size=5000,
        sgd_minibatch_size=100,
        num_sgd_iter=20,
        lr=2e-5,
    ).multi_agent(
        policies=policies,
        policy_mapping_fn=(lambda agent_id: policy_mapping[agent_id]),
    ).rollouts(
        num_rollout_workers=0,
        rollout_fragment_length=100,
    )
    # config.simple_optimizer = True

    print(config.to_dict())
    # Build an Algorithm object from the config and run 1 training iteration.
    ppo = config.build(env=env_name)
    if ckpt is not None:
        ppo.restore(os.path.join(
            checkpoints_path, f"ppo{'_parallel' if parallel else ''}-{ckpt}", f"checkpoint_{ckpt:06d}"))

    return ppo, policy_mapping


def train(ppo, cur_ckpt, episode=500, parallel=False):
    if cur_ckpt is None:
        cur_ckpt = 0

    for _ in range(1, episode + 1):
        result = ppo.train()
        print(pretty_print(result))

        if _ % 10 == 0:
            checkpoint = ppo.save(os.path.join(
                checkpoints_path, f"ppo{'_parallel' if parallel else ''}-{cur_ckpt + _}"))
            print("checkpoint saved at", checkpoint)
    return ppo, cur_ckpt + episode


def test(ppo, cur_ckpt, config_path: str = None, record=True, max_steps=500, policy=None, parallel=False):
    if config_path is None:
        env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG, max_steps=max_steps)
    else:
        env = stunning_robots.parallel_env(**stunning_robots.load_map_config(os.path.join(os.path.dirname(
            __file__), 'stunning_robots', 'maps', config_path)), auto_render=True, max_steps=max_steps)

    env = PettingZooParallelToRlLibMultiAgentEnv(env)
    observations, rewards, dones, infos = None, None, None, None
    observations = env.reset()

    frames: List[np.ndarray] = list()

    while not dones or not dones['__all__']:
        env.render()
        if record:
            frames.append(env.render('rgb_array'))

        actions = {agent: ppo.compute_single_action(
            observations[agent], policy_id=policy[agent]
        ) for agent in observations.keys()}

        observations, rewards, dones, infos = env.step(actions)

    if record:
        if not os.path.exists("video"):
            os.makedirs("video")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # opencv3.0
        h, w, c = frames[0].shape
        saved_path = os.path.join("video", f"ppo{'_parallel' if parallel else ''}-{cur_ckpt}.avi")
        video_writer = cv2.VideoWriter(saved_path, fourcc, 10, (w, h))

        for frame in frames:
            video_writer.write(np.flip(frame, 2))

        video_writer.release()
        print(f"video saved at {saved_path}")

    return ppo


def find_checkpoints(parallel=False):
    dirs = os.listdir(checkpoints_path)
    ckpt = None
    prefix = f"ppo{'_parallel' if parallel else ''}-"
    for item in dirs:
        if item.startswith(prefix):
            ckpt = int(item[len(prefix):]) if ckpt is None else max(ckpt, int(item[len(prefix):]))
    return ckpt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train PPO to play in stunning-robots.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
    parser.add_argument("-r", "--record", action='store_true')
    parser.add_argument("-p", "--parallel", action='store_true')
    parser.add_argument("-m", "--map", type=str, default=None, help="your custom map xlsx file")
    parser.add_argument("-s", "--step", type=int, default=500, help="the number of steps of an episode")
    parser.add_argument("-e", "--episode", type=int, default=500, help="the number of episodes to train")
    args = parser.parse_args()

    ray.init(log_to_driver=False)

    checkpoints = find_checkpoints(
        parallel=args.parallel)
    ppo_algorithms, mapping = build(
        checkpoints,
        config_path=args.map,
        max_steps=args.step,
        parallel=args.parallel)
    if args.command == "train":
        train(
            ppo_algorithms,
            checkpoints,
            episode=args.episode,
            parallel=args.parallel)
    elif args.command == "test":
        test(
            ppo_algorithms,
            checkpoints,
            record=args.record,
            config_path=args.map,
            max_steps=args.step,
            parallel=args.parallel,
            policy=mapping)
    else:
        parser.print_help()
