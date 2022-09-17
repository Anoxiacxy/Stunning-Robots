import functools

import gym
import pygame
import numpy as np

from typing import List, Dict, Tuple, Any, Optional
from gym import spaces

from .config import *
from .multi_robots_pz import StunningRobots


class StunningRobotsGym(StunningRobots, gym.Env):
    pass


