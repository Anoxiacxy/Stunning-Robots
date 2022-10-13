import os
import ray
from typing import List

import stunning_robots
import argparse
from ray.rllib.algorithms.a2c import A2CConfig
from ray.tune.registry import register_env

from stunning_robots.logger import RolloutLogger
from wrapper import PettingZooParallelToRlLibMultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from model.action_mask_model import TorchActionMaskModel

checkpoints_path = 'checkpoints'


def build(args, ckpt: int = None, config_path: str = None, max_steps=500, policy_name="single", render=False):
    env_name = "stunning_robots"

    def env_creator():
        if config_path is None:
            _env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG,
                                                auto_render=render, max_steps=max_steps)
        else:
            _env = stunning_robots.parallel_env(**stunning_robots.load_map_config(os.path.join(os.path.dirname(
                __file__), 'stunning_robots', 'maps', config_path)), auto_render=render, max_steps=max_steps)
        # _env = stunning_robots.aec_env(**stunning_robots.DEFAULT_CONFIG, max_steps=10000)
        return _env

    test_env = env_creator()
    periodicity: List = list(test_env.agents_periodicity.astype(int))
    agent_name_mapping = test_env.agent_name_mapping
    test_env = PettingZooParallelToRlLibMultiAgentEnv(test_env)
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

    def gen_policy(_):
        return (
            None, obs_space, act_space,
            {
                "model": {"custom_model": "action_mask_model"},
                "gamma": 0.99,
            }
        )
    if policy_name == 'parallel':
        policies = {f"{_}": gen_policy(_) for _ in range(num_agents)}
        policy_mapping = dict(zip(sorted(test_env.agent_ids), sorted(policies.keys())))
    elif policy_name == 'single':
        policies = {"policy_0": gen_policy(0)}  # "policy_0"
        policy_mapping = {agent: "policy_0" for agent in test_env.agent_ids}  # "policy_0"
    elif policy_name == 'speed':
        periodicity_set = set(periodicity)
        policies = {f"{_}": gen_policy(_) for _ in periodicity_set}
        policy_mapping = {agent: f"{periodicity[agent_name_mapping[agent]]}" for agent in test_env.agent_ids}
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    print(f"{policy_name} policy with mapping: {policy_mapping}")

    config = A2CConfig().framework(
        framework="torch"
    ).training(
        use_gae=True,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        train_batch_size=5000,
        lr=2e-5,
    ).multi_agent(
        policies=policies,
        policy_mapping_fn=(lambda agent_id: policy_mapping[agent_id]),
    ).rollouts(
        rollout_fragment_length=100,
    )
    # config.simple_optimizer = True

    # print(config.to_dict())
    a2c = config.build(env=env_name)
    if ckpt is not None:
        a2c.restore(os.path.join(
            checkpoints_path, f"a2c_{policy_name}-{ckpt}", f"checkpoint_{ckpt:06d}"))

    return a2c, policy_mapping


def train(args, a2c, cur_ckpt, episode=500, policy_name="single", checkpoint_every=10):
    if cur_ckpt is None:
        cur_ckpt = 0

    for _ in range(1, episode + 1):
        result = a2c.train()
        print(pretty_print(result))

        if _ % checkpoint_every == 0:
            checkpoint = a2c.save(os.path.join(
                checkpoints_path, f"a2c_{policy_name}-{cur_ckpt + _}"))
            print("checkpoint saved at", checkpoint)
    return a2c, cur_ckpt + episode


def test(args, a2c, cur_ckpt, config_path: str = None, record=True,
         max_steps=500, policy_mapping=None, policy_name="single", manual=None):
    if config_path is None:
        env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG, max_steps=max_steps)
    else:
        env = stunning_robots.parallel_env(**stunning_robots.load_map_config(os.path.join(os.path.dirname(
            __file__), 'stunning_robots', 'maps', config_path)), max_steps=max_steps)

    logger = RolloutLogger(env.max_num_agents, f"a2c_{policy_name}", cur_ckpt, record)
    if manual is not None and isinstance(manual, int) and 0 <= manual < env.max_num_agents:
        manual_policy = stunning_robots.ManualPolicy(env, agent_id=manual)
    else:
        manual_policy = None

    env = PettingZooParallelToRlLibMultiAgentEnv(env)
    observations, rewards, dones, infos, actions, frame = None, None, None, None, None, None
    observations = env.reset()
    env.render()
    if record:
        frame = env.render('rgb_array')
    logger.add_step(observations, rewards, dones, infos, actions, frame)

    while not dones or not dones['__all__']:
        actions = {agent: a2c.compute_single_action(
            observations[agent], policy_id=policy_mapping[agent]
        ) for agent in observations.keys()}

        if manual_policy is not None:
            actions[manual_policy.agent] = manual_policy(observations[manual_policy.agent], manual_policy.agent)

        observations, rewards, dones, infos = env.step(actions)
        env.render()
        if record:
            frame = env.render('rgb_array')
        logger.add_step(observations, rewards, dones, infos, actions, frame)

    logger.release()
    return a2c


def find_latest_checkpoint(args, policy="single"):
    dirs = os.listdir(checkpoints_path)
    ckpt = None
    prefix = f"a2c_{policy}-"
    for item in dirs:
        if item.startswith(prefix):
            ckpt = int(item[len(prefix):]) if ckpt is None else max(ckpt, int(item[len(prefix):]))
    return ckpt


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO to play in stunning-robots.')
    parser.add_argument("phase", metavar="<phase>", help="'train' or 'test'")
    parser.add_argument("-r", "--record", action="store_true", help="whether to record the test result")
    parser.add_argument("-R", "--render", action="store_true", help="whether to render during the train")
    parser.add_argument("-p", "--policy", type=str, default="single", help="'parallel', 'single', 'speed'")
    parser.add_argument("-c", "--checkpoint", type=int, default=None, help="load which checkpoint")
    parser.add_argument("-m", "--map", type=str, default=None, help="your custom map xlsx file")
    parser.add_argument("-M", "--manual", type=int, default=None, help="manually operate one specified agent")
    parser.add_argument("-s", "--step", type=int, default=500, help="the number of steps of an episode")
    parser.add_argument("-e", "--episode", type=int, default=500, help="the number of episodes to train")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    args = parser.parse_args()

    ray.init(log_to_driver=False)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = find_latest_checkpoint(args, policy=args.policy)

    a2c_algorithms, policy_mapping = build(
        args,
        checkpoint,
        config_path=args.map,
        max_steps=args.step,
        policy_name=args.policy,
        render=args.render,
    )

    if args.phase == "train":
        train(
            args,
            a2c_algorithms,
            checkpoint,
            episode=args.episode,
            policy_name=args.policy,
            checkpoint_every=args.checkpoint_every
        )
    elif args.phase == "test":
        test(
            args,
            a2c_algorithms,
            checkpoint,
            manual=args.manual,
            record=args.record,
            config_path=args.map,
            max_steps=args.step,
            policy_name=args.policy,
            policy_mapping=policy_mapping)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
