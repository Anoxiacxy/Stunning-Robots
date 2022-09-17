import stunning_robots


def parallel_mode():
    env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG)
    # average_total_reward(env, max_episodes=100, max_steps=100000)
    observations = env.reset()
    manual_policy = stunning_robots.ManualPolicy(env, agent_id=5)

    while True:
        env.render()
        actions = {agent: env.action_space(agent).sample() for agent in observations.keys()}
        if manual_policy.agent in observations.keys():
            actions[manual_policy.agent] = manual_policy(observations[manual_policy.agent], manual_policy.agent)

        observations, rewards, dones, infos = env.step(actions)
        if all(dones.values()):
            observations = env.reset()


def aec_mode():
    env = stunning_robots.aec_env(**stunning_robots.DEFAULT_CONFIG)
    env.reset()
    manual_policy = stunning_robots.ManualPolicy(env, agent_id=5)
    for agent in env.agent_iter():

        if agent == env.agents[0]:
            env.render()
        observation, reward, done, info = env.last()
        if done:
            env.step(None)
            continue

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample(observation['action_mask'])

        env.step(action)


if __name__ == "__main__":
    parallel_mode()
    # aec_mode()
