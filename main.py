import rlgym
from rlgym.gym import Gym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.reward_functions.common_rewards import AlignBallGoal
from Agents.ActorCritic import ActorCritic
from Models.BaseModel import BaseModel


def get_info(env: Gym):
    input_size = env.observation_space.shape
    print(input_size)
    action_number = env.action_space.shape

    return input_size[0], action_number[0]


def train():
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 300

    seconds = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    env = rlgym.make(game_speed=1, spawn_opponents=True,
                     terminal_conditions=[common_conditions.TimeoutCondition(seconds)],
                     reward_fn=AlignBallGoal(),
                     obs_builder=AdvancedObs())

    input_shape, action_shape = get_info(env)

    agent = ActorCritic(env)
    agent.set_model(BaseModel(input_shape, action_shape))

    for episode in range(agent.episode_number):
        obs = env.reset()
        done = False

        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = [1, 0, 1, 0, 0, 0, 0, 0]

            next_obs, reward, done, gameinfo = env.step(action)

            obs = next_obs


def main():
    train()


if __name__ == "__main__":
    main()