import os
from typing import List
from datetime import datetime
import cv2
import numpy as np
from .utils import position_to_coordinate, save_rollout_xlsx
from .config import OTHER_TYPE, ACTION_MAP_STR


class RolloutLogger:
    def __init__(self,
                 num_agents,
                 algorithm,
                 cur_ckpt,
                 record, **kwargs):
        self.coordinate_col = None
        self.coordinate_row = None
        self.step_count = None
        self.info = None
        self.reward = None
        self.action = None
        self.turn = None
        self.frames = None
        self.num_agents = None
        self.record = None
        self.cur_ckpt = None
        self.algorithm = None

        self.reset(num_agents, algorithm, cur_ckpt, record, **kwargs)

    def reset(self,
              num_agents=None,
              algorithm=None,
              cur_ckpt=None,
              record=None, **kwargs):

        if algorithm is not None:
            self.algorithm = algorithm
        if cur_ckpt is not None:
            self.cur_ckpt = cur_ckpt
        if record is not None:
            self.record = record
        if num_agents is not None:
            self.num_agents = num_agents

        # bool 是否轮到某个智能体行动
        self.turn = {f"agent_{_}": [] for _ in range(self.num_agents)}
        # HOLD, LEFT, RIGHT, UP, DOWN
        self.action = {f"agent_{_}": [] for _ in range(self.num_agents)}
        # (int, int) 智能体的坐标
        self.coordinate_row = {f"agent_{_}": [] for _ in range(self.num_agents)}
        # (int, int) 智能体的坐标
        self.coordinate_col = {f"agent_{_}": [] for _ in range(self.num_agents)}
        # int 智能体本次的奖励
        self.reward = {f"agent_{_}": [] for _ in range(self.num_agents)}
        # dict 信息 'Out of boundary', 'Peer collision', 'Goal'
        self.info = {f"agent_{_}": [] for _ in range(self.num_agents)}

        # (h, w, c)
        self.frames: List[np.ndarray] = list()
        self.step_count = 0

    def add_step(self, observation, reward, done, info, action, frame):
        if self.record:
            assert frame is not None
            self.frames.append(frame)

        assert observation is not None
        for agent in observation.keys():
            action_mask = observation[agent]['action_mask']
            obs = observation[agent]['obs']
            self.turn[agent].append(action_mask.all())
            self.action[agent].append(ACTION_MAP_STR[action[agent]] if action is not None else None)
            coordinate = position_to_coordinate(obs[..., 1 + OTHER_TYPE]) + 1
            self.coordinate_row[agent].append(coordinate[0])
            self.coordinate_col[agent].append(coordinate[1])
            self.reward[agent].append(reward[agent] if reward is not None else None)
            self.info[agent].append(info[agent] if info is not None else {})

        self.step_count += 1

    def release(self):
        now = datetime.now().__str__()
        if self.record:
            if not os.path.exists("video"):
                os.makedirs("video")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # opencv3.0
            h, w, c = self.frames[0].shape
            saved_path = os.path.join("video", f"{self.algorithm}-{self.cur_ckpt}-{now}.avi")
            video_writer = cv2.VideoWriter(saved_path, fourcc, 10, (w, h))

            for frame in self.frames:
                video_writer.write(np.flip(frame, 2))

            video_writer.release()
            print(f"video saved at {saved_path}")

        data = {f"agent_{_}": [
            ["是否可行动", "上一步的行动", "行坐标", "列坐标", "当前奖励",
             "墙体碰撞奖励", "同伴碰撞奖励", "目标点奖励", "时间惩罚", "势能奖励", "备注"]
        ] for _ in range(self.num_agents)}

        for _ in range(self.step_count):
            for i in range(self.num_agents):
                agent = f"agent_{i}"
                data[agent].append([])
                data[agent][-1].append(self.turn[agent][_])
                data[agent][-1].append(self.action[agent][_])
                data[agent][-1].append(self.coordinate_row[agent][_])
                data[agent][-1].append(self.coordinate_col[agent][_])
                data[agent][-1].append(self.reward[agent][_])
                data[agent][-1].append(
                    self.info[agent][_]['Wall collision'][-1] if 'Wall collision' in self.info[agent][_] else None)
                data[agent][-1].append(
                    self.info[agent][_]['Peer collision'][-1] if 'Peer collision' in self.info[agent][_] else None)
                data[agent][-1].append(
                    self.info[agent][_]['Goal'][-1] if 'Goal' in self.info[agent][_] else None)
                data[agent][-1].append(
                    self.info[agent][_]['Time'][-1] if 'Time' in self.info[agent][_] else None)
                data[agent][-1].append(
                    self.info[agent][_]['Potential'][-1] if 'Potential' in self.info[agent][_] else None)
                data[agent][-1].append(str(self.info[agent][_]))

        if not os.path.exists("rollout"):
            os.makedirs("rollout")
        saved_path = os.path.join("rollout", f"{self.algorithm}-{self.cur_ckpt}-{now}.xlsx")
        save_rollout_xlsx(saved_path, data)

        print(f"rollout data saved at {saved_path}")
