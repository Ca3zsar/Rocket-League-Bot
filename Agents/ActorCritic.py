import numpy as np
from rlgym.gym import Gym

from Agents.AgentBase import AgentBase


class ActorCritic(AgentBase):
    def __init__(self, env: Gym):
        super().__init__(env)

    def after_action(self):
        self.frames += 1
        self.decrease_epsilon()

        if self.num_in_buffer >= self.batch_size * 2:
            self.training()

    def training(self):
        chosen_records = self.sample()
        current = [self.current_states[index] for index in chosen_records]
        new = [self.new_states[index] for index in chosen_records]
        actions = [self.actions[index] for index in chosen_records]
        rewards = [self.rewards[index] for index in chosen_records]
        done = [self.rewards[index] for index in chosen_records]

        self.process_outputs(current, new, actions, rewards, done)

    def process_outputs(self, current, new, actions, rewards, done):
        predictions = self.target_model.predict(new)
