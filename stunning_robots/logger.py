import os
from typing import List
from datetime import datetime
import cv2
import numpy as np
from utils import coordinate_to_position, position_to_coordinate

class RolloutLogger:
    def __init__(self, **kwargs):
        self.frames = None
        self.num_agents = None
        self.record = None
        self.cur_ckpt = None
        self.algorithm = None
        self.reset(**kwargs)

    def reset(self,
              num_agents=None,
              algorithm=None,
              cur_ckpt=None,
              record=None):

        self.algorithm = algorithm
        self.cur_ckpt = cur_ckpt
        self.record = record
        self.num_agents = num_agents

        observations, rewards, dones, infos = [], [], [], []
        self.frames: List[np.ndarray] = list()

    def add_step(self, observation, reward, done, info, frame):
        if self.record:
            assert frame is not None
            self.frames.append(frame)

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