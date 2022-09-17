from typing import Dict

import pettingzoo
import gym.spaces as spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv


@PublicAPI
class PettingZooParallelToRlLibMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env: pettingzoo.ParallelEnv, obs_only=False):
        super().__init__()
        self.par_env = env
        self.par_env.reset()
        # TODO (avnishn): Remove this after making petting zoo env compatible with
        #  check_env.
        self._skip_env_checking = True

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.par_env.observation_space(self.par_env.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space = self.par_env.action_space(self.par_env.agents[0])

        assert all(
            self.par_env.observation_space(agent) == self.observation_space
            for agent in self.par_env.agents
        ), (
            "Observation spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_observations wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_observations(env)`"
        )

        assert all(
            self.par_env.action_space(agent) == self.action_space
            for agent in self.par_env.agents
        ), (
            "Action spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_action_space wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_action_space(env)`"
        )

        self.agent_ids = set(self.par_env.agents)
        self.obs_only = obs_only
        if obs_only:
            assert isinstance(self.observation_space, spaces.Dict)
            self.observation_space = self.observation_space['obs']

        else:
            sampled_state = self.state()

            assert isinstance(sampled_state, np.ndarray)
            low = sampled_state.min()
            high = sampled_state.max()

            assert isinstance(self.observation_space, spaces.Dict)
            self.observation_space[ENV_STATE] = spaces.Box(shape=sampled_state.shape, low=low, high=high)

    def reset(self):
        obss = self.par_env.reset()
        if self.obs_only:
            for agent in obss.keys():
                obss[agent] = obss[agent]['obs']
        else:
            state = self.state()
            for agent in obss.keys():
                obss[agent][ENV_STATE] = state

        return obss

    def state(self) -> np.ndarray:
        state = self.par_env.state()
        if isinstance(state, Dict):
            state = state[ENV_STATE]
        return state

    def step(self, action_dict):
        obss, rews, dones, infos = self.par_env.step(action_dict)
        self.par_env.agents = self.par_env.possible_agents
        env_is_done = all(dones.values())
        dones = {agent: env_is_done for agent in dones.keys()}
        dones["__all__"] = env_is_done

        if self.obs_only:
            for agent in obss.keys():
                obss[agent] = obss[agent]['obs']
        else:
            state = self.state()
            for agent in obss.keys():
                obss[agent][ENV_STATE] = state

        return obss, rews, dones, infos

    def close(self):
        self.par_env.close()

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def render(self, mode="human"):
        return self.par_env.render(mode)

    @property
    def unwrapped(self):
        return self.par_env.unwrapped
