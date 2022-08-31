from gym import envs
import gym_extension

env_ids = sorted([spec.id for spec in envs.registry.all()])
for _ in env_ids:
    print(_)