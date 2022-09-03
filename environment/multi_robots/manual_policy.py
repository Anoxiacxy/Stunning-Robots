import numpy as np
import pygame

from .config import *


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):

        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.possible_agents[self.agent_id]

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        if True:
            self.default_action = HOLD
            self.action_mapping = dict()
            self.action_mapping[pygame.K_a] = LEFT
            self.action_mapping[pygame.K_d] = RIGHT
            self.action_mapping[pygame.K_w] = UP
            self.action_mapping[pygame.K_s] = DOWN

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
                agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."
        # set the default action
        action = self.default_action
        if observation["action_mask"].all():
            # if we get a key, override action using the dict
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # escape to end
                        exit()

                    elif event.key == pygame.K_BACKSPACE:
                        # backspace to reset
                        self.env.reset()

                elif event.type == pygame.TEXTINPUT:
                    if ord(event.text) in self.action_mapping:
                        action = self.action_mapping[ord(event.text)]
        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping


def main():
    from pettingzoo.butterfly import pistonball_v6

    clock = pygame.time.Clock()

    env = pistonball_v6.env()
    env.reset()

    manual_policy = pistonball_v6.ManualPolicy(env)

    for agent in env.agent_iter():
        clock.tick(env.metadata["render_fps"])

        observation, reward, done, info = env.last()

        if agent == manual_policy.agent:
            action = manual_policy(observation, agent)
        else:
            action = env.action_space(agent).sample()

        action = np.array(action)
        action = action.reshape(
            1,
        )
        env.step(action)

        env.render()

        if done:
            env.reset()


if __name__ == "__main__":
    main()
