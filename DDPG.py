import numpy as np
import rlgym
import tensorflow as tf
import tensorflow.keras.initializers
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import TouchBallReward
from rlgym.utils.terminal_conditions import common_conditions
from tensorflow.keras import layers

default_tick_skip = 8
physics_ticks_per_second = 60
ep_len_seconds = 300

seconds = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))
env = rlgym.make(game_speed=50, spawn_opponents=True,
                 terminal_conditions=[common_conditions.TimeoutCondition(seconds),
                                      common_conditions.GoalScoredCondition()],
                 reward_fn=TouchBallReward(),
                 obs_builder=AdvancedObs())

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = np.asarray(obs_tuple[1]).reshape((num_actions,))
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(128, activation="relu", kernel_initializer=tensorflow.keras.initializers.Orthogonal())(inputs)
    hidden = layers.Dropout(0.2)(out)
    out = layers.Dense(256, activation="relu", kernel_initializer=tensorflow.keras.initializers.HeUniform())(hidden)
    hidden = layers.Dropout(0.3)(out)
    outputs = layers.Dense(num_actions, activation="tanh")(hidden)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(128, activation="relu", kernel_initializer=tensorflow.keras.initializers.Orthogonal())(
        state_input)
    state_out = layers.Dense(256, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu", kernel_initializer=tensorflow.keras.initializers.Orthogonal())(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(128, activation="relu")(concat)
    hidden = layers.Dropout(0.2)(out)
    out = layers.Dense(256, activation="relu")(hidden)
    hidden = layers.Dropout(0.2)(out)
    outputs = layers.Dense(num_actions)(hidden)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]


std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(num_actions))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.9
actor_lr = 0.95

critic_optimizer = tf.keras.optimizers.Adam(critic_lr, clipnorm=0.75)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr, clipnorm=0.75)

total_episodes = 10000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

update_steps = 10000
epsilon = 1
eps_decay = 0.99999

# Takes about 4 min to train
frames = 0
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    action = np.zeros((8,))

    while True:
        frames += 1
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        if frames % 4 == 0:
            prob = np.random.rand()
            if prob < epsilon:
                action = env.action_space.sample()
            else:
                action = policy(tf_prev_state, ou_noise)

        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        if reward == 0:
            reward = -0.25
        else:
            reward = 50

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        if epsilon > 0.02:
            epsilon *= eps_decay

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {} . Frames : {}".format(ep, episodic_reward, frames))
    print(f"Epsilon : {epsilon}")
    avg_reward_list.append(avg_reward)

    if ep % 100 == 0:
        actor_model.save(f"saved_models\\{ep}-actor")
        critic_model.save(f"saved_models\\{ep}-critic")
        target_actor.save(f"saved_models\\{ep}-target-actor")
        target_critic.save(f"saved_models\\{ep}-target-critic")
