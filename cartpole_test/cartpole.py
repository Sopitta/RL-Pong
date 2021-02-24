import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent_k import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)
    env.seed(1)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]
    #print(env.action_space)
    #print(env.observation_space)
    # Instantiate agent and its policy
    #policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent()

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    total_timesteps = 0
   
    agent.init_memory(1000)
    
    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        #print(observation)
        # Loop until the episode is over
        while not done:
            
            # Get action from the agent
            with torch.no_grad():
                action = agent.get_action(observation)
                #print(action)
                logprob, _ = agent.evaluate(observation, action)
                
                
            #print(action)
            next_state, reward, done, info = env.step(action.cpu().numpy())
            agent.store_outcome(total_timesteps, observation, action, logprob, reward, done)
            
            observation = next_state

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            total_timesteps += 1
            
            if (total_timesteps)%1000==0 :
                agent.update_policy(100)
                total_timesteps = 0
                

        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        #agent.update_policy(episode_number)
            

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("AC reward history (episodic)")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Episodic AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=2000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

