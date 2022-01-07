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
        self.epsilon_decay = 0.9999

        self.batch_size = 64
        self.learning_rate = 0.001
        self.record_size = 10000
        self.n_index = 0

        self.episode_number = 5000
        self.records = np.empty((self.record_size, 4))

        self.update_target_steps = 10000

        self.online_model = None
        self.target_model = None
        self.environment = env

        self.frames = 0

    def set_model(self, model: BaseModel):
        self.online_model = model
        self.target_model = model

    def add_record(self, current_state, next_state, reward, done):
        self.records[self.n_index] = np.array([current_state, next_state, reward, done])
        self.n_index = (self.n_index + 1) % self.record_size

    def get_next_action(self):
        probability = np.random.rand()

        if probability < self.epsilon:
            return self.environment.action_space.sample()

        # TODO : Update return value
        return 0

    # take 64 different action
    def sample(self):
        return np.random.choice(self.records, self.batch_size, replace=False)

    def training(self):
        pass

    def update_target(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def load_info(self, episode):
        self.online_model = tensorflow.keras.models.load_model(f"saved_models\\{episode}-online")
        self.target_model = tensorflow.keras.models.load_model(f"saved_models\\{episode}-target")

        with open(f"records\\episode_records_{episode}", "rb") as file:
            self.records = pickle.load(file)

        with open(f"configs\episode_config_{episode}.txt") as file:
            info = file.read().split()
            self.epsilon = float(info[0])
            self.frames = int(info[1])

    def serialize(self, episode):
        self.online_model.save(f'saved_models\\{episode}-online')
        self.target_model.save(f'saved_models\\{episode}-target')

        with open(f"records\\episode_records_{episode}", "wb") as file:
            pickle.dump(self.records, file, 0)

        with open(f"configs\\episode_config_{episode}.txt", "w") as file:
            file.write(f"{self.epsilon} {self.frames}")
