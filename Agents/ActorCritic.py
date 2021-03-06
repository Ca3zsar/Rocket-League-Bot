import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from rlgym.gym import Gym
from tensorflow.keras import Input, optimizers
from tensorflow.python.keras.layers import Dense, Lambda, Dropout
from tensorflow.python.keras.models import Model

from Agents.AgentBase import AgentBase

tensorflow.compat.v1.disable_eager_execution()


class ActorCritic(AgentBase):
    def __init__(self, env: Gym):
        super().__init__(env)
        self.actor, self.critic = self.build_model()
        self.optimizers = [self.actor_optimizer(), self.critic_optimizer()]

    def after_action(self):
        self.frames += 1
        self.decrease_epsilon()

        if self.num_in_buffer >= self.batch_size * 2:
            self.training()

    def training(self):
        chosen_records = self.sample()
        current = np.array([np.array(self.current_states[index]) for index in chosen_records])
        new = [self.new_states[index] for index in chosen_records]
        actions = [self.actions[index] for index in chosen_records]
        rewards = [self.rewards[index] for index in chosen_records]
        done = [self.rewards[index] for index in chosen_records]

        states = np.asarray(current, dtype='float32')
        discounted_rewards, values = self.discount_rewards(rewards, states)

        rewards = np.asarray([np.array([reward] * self.environment.action_space.shape[0]) for reward in rewards],
                             dtype='float32')

        advantages = rewards - values
        advantages = np.asarray(advantages, dtype='float32')
        actions = np.asarray(actions, dtype='float32')
        self.optimizers[0]([states, advantages, actions])
        self.optimizers[1]([states, discounted_rewards])

    def discount_rewards(self, rewards, states, done=False):
        discounted_rewards = np.zeros((len(rewards), self.environment.action_space.shape[0]))
        running_add = 0
        if not done:
            states = np.asarray(states)
            running_add = \
                self.critic.predict(states)

        for t in reversed(range(0, len(rewards))):
            discounted_rewards[t] = running_add[t] + self.gamma * rewards[t]
        return discounted_rewards, running_add

    def get_action(self, state):
        probability = np.random.rand()

        if probability < self.epsilon:
            return self.environment.action_space.sample()

        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.environment.observation_space.shape[0]]))

        action = mu + np.sqrt(sigma_sq)
        action = np.clip(action, -1, 1)
        return action

    def build_model(self):

        state = Input(batch_shape=(None, self.environment.observation_space.shape[0]))
        dropout_input = Dropout(0.2)(state)
        actor_input = Dense(128, activation='relu')(dropout_input)
        dropout_hidden = Dropout(0.4)(actor_input)
        actor_hidden = Dense(64, activation='relu')(dropout_hidden)
        dropout_hidden = Dropout(0.3)(actor_hidden)
        actor_hidden = Dense(32, activation='relu')(dropout_hidden)
        dropout_hidden = Dropout(0.1)(actor_hidden)
        mu_0 = Dense(self.environment.action_space.shape[0], activation='tanh')(dropout_hidden)
        sigma_0 = Dense(self.environment.action_space.shape[0], activation='softplus')(dropout_hidden)

        mu = Lambda(lambda x: x)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(64, input_dim=self.environment.observation_space.shape[0], activation='relu')(state)
        value_hidden = Dense(32, activation='relu', kernel_initializer='he_uniform')(critic_input)
        state_value = Dense(8, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss='mse',
                      metrics=['accuracy'])

        critic.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                       loss='mse',
                       metrics=['accuracy'])

        return actor, critic

    def actor_optimizer(self):

        action = K.placeholder(shape=(self.batch_size, 8), dtype='float32', name='actionPlaceholder')
        advantages = K.placeholder(shape=(self.batch_size, 8), dtype='float32', name='advantagesPlaceholder')

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = optimizers.Adam(learning_rate=0.0001)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=actor_loss)

        train = K.function([self.actor.input, advantages, action], [self.actor.output], updates=updates)
        return train

        # make loss function for Value approximation

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(self.batch_size, 8))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = optimizers.Adam(learning_rate=0.001)
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        train = K.function([self.critic.input, discounted_reward], [self.critic.output], updates=updates)
        return train

    def serialize(self, episode):
        self.actor.save(f'saved_models\\{episode}-online')
        self.critic.save(f'saved_models\\{episode}-target')

        # with open(f"records\\episode_records_{episode}", "wb") as file:
        #     pickle.dump(self.records, file, 0)

        with open(f"configs\\episode_config_{episode}.txt", "w") as file:
            file.write(f"{self.epsilon} {self.frames}")

    def load_info(self, episode):
        self.actor = tensorflow.keras.models.load_model(f"saved_models\\{episode}-online")
        self.critic = tensorflow.keras.models.load_model(f"saved_models\\{episode}-target")

        # with open(f"records\\episode_records_{episode}", "rb") as file:
        #     self.records = pickle.load(file)

        # with open(f"configs\\episode_config_{episode}.txt") as file:
        #     info = file.read().split()
        #     self.epsilon = float(info[0])
        #     self.frames = int(info[1])
