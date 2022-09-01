import gym
from gym import spaces
import numpy as np
from utils import coordinate_to_position, position_to_coordinate
from openpyxl import load_workbook

class RobotsEnv(gym.Env):
    CELL_TYPE = 2
    BLOCK, EMPTY = range(CELL_TYPE)

    ACTION_TYPE = 5
    HOLD, LEFT, RIGHT, UP, DOWN = range(ACTION_TYPE)
    ACTION_MAP = {
        HOLD : np.array([0, 0]),
        LEFT : np.array([0, -1]),
        RIGHT: np.array([0, 1]),
        UP: np.array([-1, 0]),
        DOWN: np.array([1, 0])
    }


    def __init__(self,
                 playground,
                 robots_speed: list,
                 robots_dest: list,
                 robots_init: list,
                 gui: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.reward = None
        self.info = None
        self.n_robot: int = len(robots_speed)
        self.action_space: spaces.Space = self._get_action_space()
        self.observation_space: spaces.Space = self._get_observation_space()
        self.playground: np.ndarray = np.array(playground)
        self.gui: bool = gui
        self.eps: float = np.finfo(float).eps

        self.robots_frequency: np.ndarray = np.array([1 / speed for speed in robots_speed])
        self.robots_dest = np.array(robots_dest)  # (..., 2)
        self.robots_init = np.array(robots_init)  # (..., 2)

        self.robots_position = np.array(robots_init) # (..., 2)
        self.robots_step_count = np.array([0 for _ in range(self.n_robot)])
        self.step_count = 0
        self.time = 0

    def step(self, action_list):
        self.time = min(self.robots_frequency[i] * (self.robots_step_count[i] + 1) for i in range(self.n_robot))

        action_mask = self.time + self.eps >= self.robots_step_count * self.robots_frequency
        robots_next_position = self.robots_position.copy()

        self.info = [[] for _ in range(self.n_robot)]
        self.reward = [0 for _ in range(self.n_robot)]

        for i in range(self.n_robot):
            if action_mask[i]:
                robots_next_position[i] += self.ACTION_MAP[action_list[i].argmax()]
            elif action_list[i].argmax() != self.HOLD:
                self.info[i].append(f"Invalid action: robot-{i} at {self.robots_position[i]} "
                                    f"{self.ACTION_MAP[action_list[i].argmax()]}")
                self.reward

        # collision detection



    def reset(self):
        self.step_count = 0
        self.time = 0

    def render(self, mode='human'):
        pass

    def _get_observation(self):
        pass

    def _get_reward(self):
        pass

    def _get_done(self) -> list:
        return [self.robots_dest[i] == self.robots_position[i] for i in range(self.n_robot)]

    def _get_info(self):
        pass

    def _get_action_space(self) -> spaces.Space:
        return spaces.Tuple(
            spaces.Discrete(self.ACTION_TYPE) for _ in range(self.n_robot)
        )

    def _get_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "playground": spaces.MultiBinary(self.playground.shape),
            "robots_position": spaces.Tuple(
                spaces.MultiDiscrete(self.playground.shape) for _ in range(self.n_robot)
            ),
            "robots_goal": spaces.Tuple(
                spaces.MultiDiscrete(self.playground.shape) for _ in range(self.n_robot)
            ),
        })