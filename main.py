import numpy as np
import rlgym
from rlgym.gym import Gym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.reward_functions.common_rewards import AlignBallGoal, VelocityBallToGoalReward, \
    LiuDistancePlayerToBallReward, RewardIfTouchedLast

from rlgym_tools.extra_rewards.multiply_rewards import MultiplyRewards

from Agents.ActorCritic import ActorCritic
from Models.BaseModel import BaseModel
from Rewards.KickoffReward import KickoffReward


def get_info(env: Gym):
    input_size = env.observation_space.shape
    print(input_size)
    action_number = env.action_space.shape

    return input_size[0], action_number[0]


def train():
    default_tick_skip = 8
    physics_ticks_per_second = 60
    ep_len_seconds = 300

    seconds = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    env = rlgym.make(game_speed=100, spawn_opponents=True,
                     terminal_conditions=[common_conditions.TimeoutCondition(seconds), common_conditions.GoalScoredCondition()],
                     reward_fn=MultiplyRewards(VelocityBallToGoalReward(),LiuDistancePlayerToBallReward()),
                     obs_builder=AdvancedObs())

    agent = ActorCritic(env)

    for episode in range(agent.episode_number):
        obs = env.reset(True)[0]
        done = False

        total_reward = 0

        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = agent.get_action(obs)

            old_state = np.copy(obs)

            obs, reward, done, gameinfo = env.step(action)

            total_reward += reward

            agent.add_record(old_state, obs, action, reward, done)

            agent.after_action()
            if done:
                print(f"Episode {episode} finished with score {total_reward}")
                print(f"Epsilon for {episode} : {agent.epsilon}")

        if episode % 50 == 0:
            agent.serialize(episode)


def test(model_name):
    default_tick_skip = 8
    physics_ticks_per_second = 60
    ep_len_seconds = 300

    seconds = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    env = rlgym.make(game_speed=2, spawn_opponents=True,
                     terminal_conditions=[common_conditions.TimeoutCondition(seconds),
                                          common_conditions.GoalScoredCondition()],
                     obs_builder=AdvancedObs())

    agent = ActorCritic(env)
    agent.load_info(model_name)
    agent.epsilon = 0.01

    for episode in range(100):
        obs = env.reset(True)[0]
        done = False

        while not done:
            action = agent.get_action(obs).flatten()
            print(action)
            action[2] = 0.0
            action[3] = 0.0
            action[4] = 0.0

            obs, reward, done, gameinfo = env.step(action)

            if done:
                print(f"Episode {episode} finished with score {gameinfo}")


def main():
    train()
    # test("350")


if __name__ == "__main__":
    main()