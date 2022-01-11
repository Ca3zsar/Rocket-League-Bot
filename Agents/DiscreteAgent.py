import random

import numpy as np
from rlgym.gym import Gym

from Agents.AgentBase import AgentBase

ACTIONS = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([1, 1, 0, 0, 0, 0, 0, 0]),
    np.array([1, -1, 0, 0, 0, 0, 0, 0]),
    np.array([-1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([-1, 1, 0, 0, 0, 0, 0, 0]),
    np.array([-1, -1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0])
]

class DiscreteAgent(AgentBase):
    def __init__(self, env: Gym):
        super().__init__(env)

    def get_next_action(self, state):
        probability = np.random.rand()

        if probability < self.epsilon:
            return random.randint(0, 7)

        new_state = np.reshape(state, (1, 107))
        return np.argmax(self.online_model.model.predict(new_state))

    def after_action(self):
        self.decrease_epsilon()

        if self.frames % 8 == 0 and self.num_in_buffer > 2*self.batch_size:
            self.training()

        if self.frames % self.update_target_steps == 0:
            self.update_target()

    def training(self):
        chosen_records = self.sample()
        current = np.array([np.array(self.current_states[index]) for index in chosen_records])
        new = np.array([self.new_states[index] for index in chosen_records])
        actions = np.array([self.actions[index] for index in chosen_records])
        rewards = np.array([self.rewards[index] for index in chosen_records])
        done = np.array([self.rewards[index] for index in chosen_records])

        outputs = self.__get_outputs(current, actions, rewards, new, done)

        # i'm not sure it here it should be epochs=1
        self.online_model.model.fit(current, outputs, epochs=1, verbose=0)


    def __get_outputs(self, states, actions, rewards, new_states, done):
        new_states_reshaped = np.array(new_states)
        predictions = self.target_model.model.predict(new_states_reshaped)
        max_qs = np.max(predictions, axis=1)

        predictions = self.online_model.model.predict(states)

        for index, action in enumerate(actions):
            if not done[index]:
                predictions[index, int(action)] = rewards[index] + self.gamma * max_qs[index]
            else:
                predictions[index, int(action)] = rewards[index]

        return predictions
