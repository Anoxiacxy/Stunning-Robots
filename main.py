import numpy as np

from pettingzoo.utils import random_demo, average_total_reward
from pettingzoo.test import render_test, performance_benchmark, test_save_obs, api_test, parallel_api_test
from environment import multi_robots


def parallel_mode():
    env = multi_robots.parallel_env(**multi_robots.DEFAULT_CONFIG)
    # average_total_reward(env, max_episodes=100, max_steps=100000)
    observations = env.reset()
    manual_policy = multi_robots.ManualPolicy(env, agent_id=5)

    while True:
        env.render()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        if manual_policy.agent in actions.keys():
            actions[manual_policy.agent] = manual_policy(observations[manual_policy.agent], manual_policy.agent)

        observations, rewards, dones, infos = env.step(actions)
        if all(dones.values()):
            observations = env.reset()


def ace_mode():
    env = multi_robots.env(**multi_robots.DEFAULT_CONFIG)
    # average_total_reward(env, max_episodes=100, max_steps=100000)
    env.reset()
    manual_policy = multi_robots.ManualPolicy(env, agent_id=5)
    for agent in env.agent_iter():
        # print(len(env.agents))
        # if len(env.agents) != 10:
        #     print("gg")
        if agent == env.agents[0]:
            env.render()
        observation, reward, done, info = env.last()

        if done:
            env.step(None)
            continue

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample()

        env.step(action)


ace_mode()
