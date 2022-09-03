# https://blog.csdn.net/qq_47997583/article/details/122508079
from gym.envs.registration import register

register(
    id='Robots-v0',
    entry_point='environment.multi_robots:RobotsEnv',
)
