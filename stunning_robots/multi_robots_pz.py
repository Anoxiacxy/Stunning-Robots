import functools

import pygame

from typing import List, Dict, Tuple, Any, Optional
from gym import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

from .config import *


def aec_env(**kwargs):
    """
    AEC 模式的环境，就是每个智能体按照某个顺序依次行动
    """
    cur_env = raw_env(**kwargs)
    # This wrapper is only for environments which print results to the terminal
    cur_env = wrappers.CaptureStdoutWrapper(cur_env)
    # this wrapper helps error handling for discrete action spaces
    cur_env = wrappers.AssertOutOfBoundsWrapper(cur_env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    cur_env = wrappers.OrderEnforcingWrapper(cur_env)
    return cur_env


def parallel_env(**kwargs):
    """
    PARALLEL 模式的环境，所有的智能体在同一个时刻做出决策
    """
    return StunningRobots(**kwargs)


def raw_env(**kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    cur_env = parallel_env(**kwargs)
    cur_env = parallel_to_aec(cur_env)
    return cur_env


class StunningRobots(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "MultiRobots", 'is_parallelizable': True}

    def __init__(self,
                 grids: List[List],
                 periodicity: List,
                 goal_pos: List[List],
                 init_pos: List,
                 render_ratio: int = 16,
                 max_steps: Optional[int] = None,
                 auto_render: bool = False,
                 fps=3,
                 **kwargs):
        super().__init__()

        self.auto_render = auto_render
        self.infos = None
        self.font = None
        self.fps_clock = None
        self.has_goal = None
        self.action_turn = None
        self.rewards = None

        self.possible_agents = [f"agent_{_}" for _ in range(len(init_pos))]
        self.agents = []

        self.agent_name_mapping: Dict[str, int] = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.num_goals = len(goal_pos[0])

        self.grids: np.ndarray = np.array(grids, dtype=int)  # (h, w)
        self.height = self.grids.shape[0]
        self.width = self.grids.shape[1]

        self.action_spaces: Dict[str, spaces.Space] = self._get_action_space()
        self.observation_spaces: Dict[str, spaces.Space] = self._get_observation_space()
        self.state_space: spaces.Space = self._get_state_space()

        self.agents_periodicity: np.ndarray = np.array(periodicity)  # (n,)
        self.agents_goal_pos = np.array(goal_pos)  # (n, m, 2)
        self.agents_init_pos = np.array(init_pos)  # (n, 2)

        self.agents_pos: np.ndarray = self.agents_init_pos.copy()  # (n, 2)
        self.agents_move_step = np.array([0 for _ in range(self.num_agents)])  # (n,)
        self.agents_goal_step = np.array([0 for _ in range(self.num_agents)])  # (n,)

        self.step_count = 0

        self.render_on = False
        self.render_ratio = render_ratio
        self.fps = fps
        self.screen = None
        self.margin = 10
        self.max_steps = max_steps

    def reset(self,
              record: Optional[bool] = False,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.agents_pos = self.agents_init_pos.copy()  # (n, 2)
        self.agents_move_step = np.array([0 for _ in range(self.num_agents)])  # (n,)
        self.agents_goal_step = np.array([0 for _ in range(self.num_agents)])  # (n,)

        self.action_turn = self.step_count + EPS >= self.agents_move_step * self.agents_periodicity
        self.has_goal = self.agents_goal_step < self.num_goals
        return self._get_observations()

    def seed(self, seed=None):
        super().seed(seed)

    def step(self, actions) -> Tuple[dict, dict, Dict[str, bool], Dict[str, Any]]:
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # print(self.step_count)
        self.action_turn = self.step_count + EPS >= self.agents_move_step * self.agents_periodicity
        self.has_goal = self.agents_goal_step < self.num_goals
        self.rewards = np.zeros(self.max_num_agents)
        self.infos = [{} for _ in range(self.max_num_agents)]

        agents_next_pos = self.agents_pos.copy()

        for agent in self.agents:
            if agent not in actions:
                continue
            agent_id = self.agent_name_mapping[agent]
            if self.action_turn[agent_id]:
                agents_next_pos[agent_id] += ACTION_MAP[actions[agent]]
                # if actions[agent] == HOLD:
                #     self.rewards[agent_id] -= 1
                #     self.infos[agent_id]['Non-recommended action'] = \
                #         f"robot-{agent_id} at {self.agents_pos[agent_id]} -> {ACTION_MAP[actions[agent]]}"
            # else:
            #     if actions[agent] != HOLD:
            #         self.rewards[agent_id] -= 1
            #         self.infos[agent_id]['Invalid action'] = \
            #             f"robot-{agent_id} at {self.agents_pos[agent_id]} -> {ACTION_MAP[actions[agent]]}"

        # collision detection
        for i in range(self.max_num_agents):
            if not 0 <= agents_next_pos[i][0] < self.height:
                agents_next_pos[i] = self.agents_pos[i].copy()
                self.rewards[i] += OUT_OF_BOUNDARY_REWARD
                self.infos[i]['Out of boundary'] = (
                    f"robot-{i} at {self.agents_pos[i] + 1} -> "
                    f"{self.agents_pos[i] + ACTION_MAP[actions[self.possible_agents[i]]] + 1}",
                    OUT_OF_BOUNDARY_REWARD
                )

            if not 0 <= agents_next_pos[i][1] < self.width:
                agents_next_pos[i] = self.agents_pos[i].copy()
                self.rewards[i] += OUT_OF_BOUNDARY_REWARD
                self.infos[i]['Out of boundary'] = (
                    f"robot-{i} at {self.agents_pos[i] + 1} -> "
                    f"{self.agents_pos[i] + ACTION_MAP[actions[self.possible_agents[i]]] + 1}",
                    OUT_OF_BOUNDARY_REWARD
                )

            if not self.grids[agents_next_pos[i][0], agents_next_pos[i][1]]:
                agents_next_pos[i] = self.agents_pos[i].copy()
                self.rewards[i] += WALL_COLLISION_REWARD
                self.infos[i]['Wall collision'] = (
                    f"robot-{i} at {self.agents_pos[i] + 1} -> "
                    f"{self.agents_pos[i] + ACTION_MAP[actions[self.possible_agents[i]]] + 1}",
                    WALL_COLLISION_REWARD
                )

        while True:
            has_collision = False
            for i in range(self.max_num_agents):
                if self.action_turn[i]:
                    for j in range(self.max_num_agents):
                        if j != i and (agents_next_pos[i] == agents_next_pos[j]).all():
                            if (agents_next_pos[i] != self.agents_pos[i]).any():
                                agents_next_pos[i] = self.agents_pos[i].copy()
                                self.rewards[i] += PEER_COLLISION_REWARD
                                self.infos[i]['Peer collision'] = (
                                    f"robot-{i} collides with robot-{j} at {self.agents_pos[i] + 1} -> "
                                    f"{self.agents_pos[i] + ACTION_MAP[actions[self.possible_agents[i]]] + 1}",
                                    PEER_COLLISION_REWARD
                                )

                            if (agents_next_pos[j] != self.agents_pos[j]).any():
                                agents_next_pos[j] = self.agents_pos[j].copy()
                                self.rewards[i] += PEER_COLLISION_REWARD
                                self.infos[j]['Peer collision'] = (
                                    f"robot-{j} collides with robot-{i} at {self.agents_pos[j] + 1} -> "
                                    f"{self.agents_pos[j] + ACTION_MAP[actions[self.possible_agents[j]]] + 1}",
                                    PEER_COLLISION_REWARD
                                )

                            has_collision = True
            if not has_collision:
                break

        self.step_count += 1
        self.agents_move_step += self.action_turn
        agents_last_pos = self.agents_pos
        self.agents_pos = agents_next_pos
        for i in range(self.max_num_agents):
            j = self.agents_goal_step[i]
            if self.action_turn[i] and self.has_goal[i]:
                # 轮到该智能体行动的时候
                if (self.agents_pos[i] == self.agents_goal_pos[i][j]).all():
                    # 到达目标的奖励分数
                    self.rewards[i] += GOAL_REWARD
                    self.agents_goal_step[i] += 1
                    self.infos[i]['Goal'] = (
                        f"robot-{i} reaches goal {self.agents_goal_step[i]} at {self.agents_pos[i] + 1}",
                        GOAL_REWARD
                    )

                else:
                    # 未到达目标
                    self.infos[i]['Time'] = (
                        None,
                        TIME_PUNISHMENT
                    )
                    self.rewards[i] += TIME_PUNISHMENT
                    last_dist = np.abs(self.agents_goal_pos[i][j] - agents_last_pos[i])
                    next_dist = np.abs(self.agents_goal_pos[i][j] - agents_next_pos[i])
                    if next_dist.sum() < last_dist.sum():
                        self.rewards[i] += POTENTIAL_REWARD
                        self.infos[i]['Potential'] = (
                            None,
                            POTENTIAL_REWARD
                        )
                    elif next_dist.sum() > last_dist.sum():
                        self.rewards[i] -= POTENTIAL_REWARD
                        self.infos[i]['Potential'] = (
                            None,
                            -POTENTIAL_REWARD
                        )

        self.has_goal = self.agents_goal_step < self.num_goals
        self.action_turn = self.step_count + EPS >= self.agents_move_step * self.agents_periodicity

        observations = self._get_observations()
        rewards = self._get_rewards()
        dones = self._get_dones()
        infos = self._get_infos()

        self.agents = [agent for agent in self.agents if not dones[agent]]

        if self.step_count % 100 == 0:
            self.render("ansi")
        if self.auto_render:
            self.render(fps_limit=False)

        return observations, rewards, dones, infos

    def render(self, mode='human', fps_limit=True):
        if mode in ['human', 'rgb_array'] and \
                (not pygame.get_init() or self.screen is None):
            pygame.init()
            self.font = pygame.font.Font(pygame.font.get_default_font(), self.render_ratio * 2 // 3)
            self.screen = pygame.display.set_mode((
                self.width * self.render_ratio + self.margin * 2,
                self.height * self.render_ratio + self.margin * 2),
            )
            # flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME)
            pygame.display.set_caption('stunning_robots')
            self.fps_clock = pygame.time.Clock()

        if mode == "human":
            self._draw()
            # self.fps_clock.tick_busy_loop()
            if fps_limit:
                self.fps_clock.tick(self.fps)
            pygame.display.update()
            return None
        elif mode == 'rgb_array':
            self._draw()
            rgb = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(rgb, axes=(1, 0, 2))
        elif mode == 'ansi':
            print(f"{self.step_count} -- ", end="")
            for agent in self.agents:
                agent_id = self.agent_name_mapping[agent]
                if self.has_goal[agent_id]:
                    distance = np.abs(self.agents_pos[agent_id]
                                      - self.agents_goal_pos[agent_id][self.agents_goal_step[agent_id]])
                else:
                    distance = 0
                print(f"{agent_id}: {distance} \t", end="")
            print("")

    def _draw(self):
        self.screen.fill(white)
        # Draw grids
        for i in range(self.width + 1):
            start = (self.margin + i * self.render_ratio, self.margin,)
            end = (self.margin + i * self.render_ratio, self.margin + self.height * self.render_ratio,)
            pygame.draw.line(self.screen, gray, start, end)

        for i in range(self.height + 1):
            start = (self.margin, self.margin + i * self.render_ratio,)
            end = (self.margin + self.width * self.render_ratio, self.margin + i * self.render_ratio,)
            pygame.draw.line(self.screen, gray, start, end)

        # Draw Barriers
        for i in range(self.width):
            for j in range(self.height):
                if not self.grids[j][i]:
                    rect = pygame.Rect(
                        (self.margin + i * self.render_ratio, self.margin + j * self.render_ratio),
                        (self.render_ratio, self.render_ratio))
                    pygame.draw.rect(self.screen, black, rect=rect)

        # Draw Goals
        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]
            if self.has_goal[agent_id]:
                text = self.font.render(f"{agent_id}", True, black)
                center = (self.agents_goal_pos[agent_id][self.agents_goal_step[agent_id]]
                          * self.render_ratio + self.margin + (self.render_ratio / 2))[[1, 0]]
                pygame.draw.circle(self.screen, gold, center, self.render_ratio / 2)
                self.screen.blit(text, text.get_rect(center=center))

        # Draw Robots
        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]
            text = self.font.render(f"{agent_id}", True, black)
            center = (self.agents_pos[agent_id] * self.render_ratio + self.margin + (self.render_ratio / 2))[[1, 0]]
            pygame.draw.circle(self.screen, blue, center, self.render_ratio / 2)
            self.screen.blit(text, text.get_rect(center=center))

    def close(self):
        if pygame.get_init():
            pygame.event.pump()
            pygame.display.quit()
            pygame.quit()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        return spaces.Dict({
            "obs": spaces.MultiBinary((self.height, self.width, CELL_TYPE + SELF_TYPE + OTHER_TYPE)),
            "action_mask": spaces.MultiBinary(ACTION_TYPE)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> spaces.Space:
        return spaces.Discrete(ACTION_TYPE)

    def state(self) -> Dict:
        state = np.zeros((self.height, self.width, CELL_TYPE + self.max_num_agents * SELF_TYPE), dtype=np.int8)
        state[:, :, GRID] = self.grids
        for i in range(self.max_num_agents):
            state[self.agents_pos[i][0], self.agents_pos[i][1], i * SELF_TYPE + CELL_TYPE] = 1
            cur_goal_step = self.agents_goal_step[i]
            if self.has_goal[i]:
                state[self.agents_goal_pos[i][cur_goal_step][0],
                      self.agents_goal_pos[i][cur_goal_step][1], i * SELF_TYPE + CELL_TYPE + 1] = 1

            for j in range(cur_goal_step, self.num_goals):
                state[self.agents_goal_pos[i][j][0],
                      self.agents_goal_pos[i][j][1], i * SELF_TYPE + CELL_TYPE + 2] = 1
        return {
            "state": state,
            "action_turn": self.action_turn
        }

    def _get_observation(self, agent):
        agent_id = self.agent_name_mapping[agent]
        obs = np.zeros((self.height, self.width, CELL_TYPE + OTHER_TYPE + SELF_TYPE), dtype=np.int8)
        obs[:, :, GRID] = self.grids
        for other in self.agents:
            idx = self.agent_name_mapping[other]
            base = OTHER_AGENTS, OTHER_GOAL, OTHER_GOALS
            if other == agent:
                base = SELF_AGENT, SELF_GOAL, SELF_GOALS
            obs[self.agents_pos[idx][0], self.agents_pos[idx][1], base[0]] = 1
            if self.has_goal[idx]:
                cur_goal_step = self.agents_goal_step[idx]
                obs[self.agents_goal_pos[idx][cur_goal_step][0],
                    self.agents_goal_pos[idx][cur_goal_step][1], base[1]] = 1

                for i in range(cur_goal_step, self.num_goals):
                    obs[self.agents_goal_pos[idx][i][0],
                        self.agents_goal_pos[idx][i][1], base[2]] = 1

        mask = np.ones(ACTION_TYPE, dtype=np.int8)
        if not self.action_turn[agent_id] or not self.has_goal[agent_id]:
            mask = np.zeros(ACTION_TYPE, dtype=np.int8)
            mask[HOLD] = 1
        return {
            "obs": obs,
            "action_mask": mask
        }

    def _get_observations(self):
        return {agent: self._get_observation(agent) for agent in self.agents}

    def _get_reward(self, agent):
        agent_id = self.agent_name_mapping[agent]
        return self.rewards[agent_id]

    def _get_rewards(self):
        return {agent: self._get_reward(agent) for agent in self.agents}

    def _get_done(self, agent):
        agent_id = self.agent_name_mapping[agent]
        return (not self.has_goal[agent_id]) or \
               (self.step_count >= self.max_steps if self.max_steps is not None else False)

    def _get_dones(self) -> Dict[str, bool]:
        return {agent: self._get_done(agent) for agent in self.agents}

    def _get_info(self, agent):
        agent_id = self.agent_name_mapping[agent]
        return self.infos[agent_id]

    def _get_infos(self) -> Dict[str, Any]:
        infos = {agent: self._get_info(agent) for agent in self.agents}
        return infos

    def _get_action_space(self) -> Dict[str, spaces.Space]:
        return dict([(f"agent_{_}", self.action_space(f"agent_{_}")) for _ in range(self.num_agents)])

    def _get_observation_space(self) -> Dict[str, spaces.Space]:
        return dict([(f"agent_{_}", self.observation_space(f"agent_{_}")) for _ in range(self.num_agents)])

    def _get_state_space(self) -> spaces.Space:
        return spaces.Dict({
            "action_turn": spaces.MultiBinary(self.max_num_agents),
            "state": spaces.MultiBinary((self.height, self.width, CELL_TYPE + self.max_num_agents * SELF_TYPE))
        })
