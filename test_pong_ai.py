"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
import src.agents.ppo_agent
from src.agents import ppo_agent, ppo_agent_stack, ppo_agent_stack_2, ppo_agent_stack_3, ppo_agent_stack_4, ppo_agent_stack_5, ppo_agent_st_1

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
#env = gym.make("WimblepongVisualBadAI-v0")
#env = gym.make("WimblepongVisualSomeAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 100000

# Define the player
player_id = 1
n_actions = env.action_space.n
# Set up the player here. We used the SimpleAI that does not take actions for now
#player = wimblepong.SimpleAi(env, player_id)
#player = DQNagent.Agent(n_actions,player_id,gamma=0.98,n_stack=4)
player = ppo_agent_stack_5.Agent()
#player = ppo_agent_stack_4.Agent()
#player = ppo_agent_st_1.Agent()
player.load_model()
env.set_names(player.get_name())
observation = env.reset()
# Housekeeping
states = []
win1 = 0

for i in range(0,episodes):
    done = False
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        # action1 = 0 #player.get_action()
        action1 = player.get_action(observation)
        ob1, rew1, done, info = env.step(action1)
        observation = ob1
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            player.reset()
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            if i % 5 == 4:
                env.switch_sides()

