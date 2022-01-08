import numpy as np
import tensorflow
from rlgym.gym import Gym
from tensorflow.keras import Input, optimizers
from tensorflow.python.keras.models import Model
import tensorflow.keras.backend as K
from Agents.AgentBase import AgentBase
from tensorflow.python.keras.layers import Dense, InputLayer, Lambda

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
        current = [self.current_states[index] for index in chosen_records]
        new = [self.new_states[index] for index in chosen_records]
        actions = [self.actions[index] for index in chosen_records]
        rewards = [self.rewards[index] for index in chosen_records]
        done = [self.rewards[index] for index in chosen_records]

        # self.process_outputs(current, new, actions, rewards, done)

        # discounted_rewards = self.discount_rewards(self.rewards, done)

        # print("states size : ",len(self.states)," ", len(self.states[0]))
        # print("actions_size : ",len(self.actions))

        states = np.asarray(current, dtype='float32')
        rewards = np.asarray(rewards, dtype='float32')
        values = self.critic.predict(states)

        advantages = rewards - values

        # action = np.array(self.actions)
        # print(action.shape)
        # print(advantages.shape)

        self.optimizers[0]([self.states, self.actions, advantages])
        self.optimizers[1]([self.states, rewards])
        self.states, self.actions, self.rewards = [], [], []


    def get_action(self, state):
        probability = np.random.rand()

        if probability < self.epsilon:
            return self.environment.action_space.sample()

        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.environment.observation_space.shape[0]]))

        epsilon = np.random.randn(self.environment.action_space.shape[0])
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -1, 1)
        return action


    def process_outputs(self, current, new, actions, rewards, done):
        predictions = self.target_model.predict(new)


    def build_model(self):

        state = Input(batch_shape=(None, self.environment.observation_space.shape[0]))
        actor_input = Dense(24, input_dim=self.environment.observation_space.shape[0], activation='relu')(state)
        actor_hidden = Dense(24, activation='relu')(actor_input)
        mu_0 = Dense(self.environment.action_space.shape[0], activation='tanh')(actor_hidden)
        sigma_0 = Dense(self.environment.action_space.shape[0], activation='softplus')(actor_hidden)

        mu = Lambda(lambda x: x * 2)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(24, input_dim=self.environment.observation_space.shape[0], activation='relu')(state)
        value_hidden = Dense(24, activation='relu', kernel_initializer='he_uniform')(critic_input)
        state_value = Dense(8, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        # actor._make_predict_function()
        # critic._make_predict_function()
        actor.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss='mse',
                      metrics=['accuracy'])

        critic.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                       loss='mse',
                       metrics=['accuracy'])

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):

        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        # mu = K.placeholder(shape=(None, self.action_size))
        # sigma_sq = K.placeholder(shape=(None, self.action_size))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = optimizers.Adam(learning_rate=0.0001)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=actor_loss)

        train = K.function(self.actor.input, action, updates=updates)
        return train

        # make loss function for Value approximation

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = optimizers.Adam(learning_rate=0.001)
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        train = K.function(self.critic.input, discounted_reward, updates=updates)
        return train
