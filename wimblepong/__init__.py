import gym
from wimblepong.wimblepong import Wimblepong
from wimblepong.simple_ai import SimpleAi
from wimblepong.bad_ai import BadAi
from wimblepong.some_ai import SomeAi

gym.envs.register("WimblepongVisualMultiplayer-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": None, "visual": True})

gym.envs.register("WimblepongVisualSimpleAI-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": SimpleAi, "visual": True})

gym.envs.register("WimblepongMultiplayer-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": None, "visual": False})

gym.envs.register("WimblepongSimpleAI-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": SimpleAi, "visual": False})

#extra environment for bad ai
gym.envs.register("WimblepongVisualBadAI-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": BadAi, "visual": True})

#extra environment for some ai
gym.envs.register("WimblepongVisualSomeAI-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": SomeAi, "visual": True})