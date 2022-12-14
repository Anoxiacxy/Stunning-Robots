import os
import numpy as np
from .utils import load_map_config

CELL_TYPE = 1
SELF_TYPE = 3
OTHER_TYPE = 3
GRID, \
    OTHER_AGENTS, OTHER_GOAL, OTHER_GOALS, \
    SELF_AGENT, SELF_GOAL, SELF_GOALS \
    = range(CELL_TYPE + OTHER_TYPE + SELF_TYPE)

ACTION_TYPE = 5
HOLD, LEFT, RIGHT, UP, DOWN = range(ACTION_TYPE)
ACTION_MAP = {
    HOLD: np.array([0, 0]),
    LEFT: np.array([0, -1]),
    RIGHT: np.array([0, 1]),
    UP: np.array([-1, 0]),
    DOWN: np.array([1, 0])
}
ACTION_MAP_STR = {
    HOLD: "HOLD",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    UP: "UP",
    DOWN: "DOWN"
}

EPS: float = np.finfo(float).eps

blue = (159, 197, 232)
gray = (217, 217, 217)
gold = (255, 215, 0)
black = (0, 0, 0)
white = (255, 255, 255)

GOAL_REWARD = +100
WALL_COLLISION_REWARD = -1
PEER_COLLISION_REWARD = -1
OUT_OF_BOUNDARY_REWARD = -1
POTENTIAL_REWARD = +1.5
TIME_PUNISHMENT = -2.5

DEFAULT_CONFIG = load_map_config(os.path.join(os.path.dirname(__file__), "maps/data1.xlsx"))