from rlgym.gym import Gym

from Agents.AgentBase import AgentBase


class ActorCritic(AgentBase):
    def __init__(self, env : Gym):
        super().__init__(env)