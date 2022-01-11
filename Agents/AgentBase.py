import pickle

import numpy as np
import tensorflow.keras.models
from rlgym.gym import Gym

from Models import BaseModel

class AgentBase:
    def __init__(self, env: Gym):
        self.gamma = 0.95
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.max_epsilon = 1
        self.epsilon_decay = 0.999995

        self.batch_size = 64
        self.learning_rate = 0.001
        self.record_size = 10000
        self.n_index = 0
        self.num_in_buffer = 0

        self.episode_number = 5000
        self.current_states = np.empty((self.record_size,) + env.observation_space.shape)
        self.new_states = np.empty((self.record_size,) + env.observation_space.shape)
        self.actions = np.empty((self.record_size,))
        self.rewards = np.empty((self.record_size,))
        self.done = np.empty((self.record_size,))

        self.update_target_steps = 10000

        self.online_model: BaseModel = None
        self.target_model: BaseModel = None
        self.environment = env

        self.frames = 0

    def set_model(self, model: BaseModel):
        self.online_model = model
        self.target_model = model

    def add_record(self, current_state, next_state, action, reward, done):
        self.current_states[self.n_index] = current_state
        self.new_states[self.n_index] = next_state
        self.actions[self.n_index] = action
        self.rewards[self.n_index] = reward
        self.done[self.n_index] = done

        self.n_index = (self.n_index + 1) % self.record_size
        self.num_in_buffer = min(self.record_size, self.n_index)

    def get_next_action(self, state):
        pass


    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def after_action(self):
        pass

    # take 64 different action
    def sample(self):

        return np.random.choice(self.num_in_buffer, self.batch_size*2, replace=False)

    def training(self):
        pass

    def update_target(self):
        self.target_model.model.set_weights(self.online_model.model.get_weights())

    def load_info(self, episode):
        self.online_model.model = tensorflow.keras.models.load_model(f"saved_models\\{episode}-online")
        self.target_model.model = tensorflow.keras.models.load_model(f"saved_models\\{episode}-target")

        with open(f"records\\episode_records_{episode}", "rb") as file:
            self.records = pickle.load(file)

        with open(f"configs\\episode_config_{episode}.txt") as file:
            info = file.read().split()
            self.epsilon = float(info[0])
            self.frames = int(info[1])

    def serialize(self, episode):
        self.online_model.model.save(f'saved_models\\{episode}-online')
        self.target_model.model.save(f'saved_models\\{episode}-target')

        # with open(f"records\\episode_records_{episode}", "wb") as file:
        #     pickle.dump(self.records, file, 0)
        #
        # with open(f"configs\\episode_config_{episode}.txt", "w") as file:
        #     file.write(f"{self.epsilon} {self.frames}")
