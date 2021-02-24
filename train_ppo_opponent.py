import gym
from src.agents import ppo_agent, DQNagent, ppo_agent_st_1
import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
from distutils.util import strtobool
import wimblepong
from parallel_env import ParallelEnvs
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


matplotlib.use("Agg")


def train(env, agent, args):
    if args.verbose:
        print(f"Training Wimblepong agent with actor-critic PPO for {args.timesteps} total timesteps.")

    state = env.reset()
    reward_sum = np.zeros(args.num_envs)
    episode_rewards = []
    avg_episode_rewards = []
    steps_delta = np.zeros(args.num_envs)
    episode_lengths = []
    avg_episode_lengths = []
    episode_count = 0
    updates = args.timesteps // args.batch_size
    global_step = 0
    for i in range(1, updates+1):
        for step in range(args.steps_per_env):
            global_step += args.num_envs
            steps_delta += np.array([1 for _ in range(args.num_envs)])
            with torch.no_grad():
                action = agent.get_action(state)
                logprob, _ = agent.evaluate(state, action)

            next_state, reward, done, info = env.step(action.cpu())

            reward_sum += reward
            zero_inds = reward == 0

            if args.survival_bonus:
                reward[zero_inds] = 1

            agent.store_outcome(step, state, action, logprob, reward, done)

            for j in range(args.num_envs):
                if done[j]:
                    episode_rewards.append(np.mean(reward_sum[j]))
                    reward_sum[j] = 0
                    episode_lengths.append(steps_delta[j])
                    steps_delta[j] = 0
                    episode_count += 1
                    if episode_count > 100:
                        avg = np.mean(episode_rewards[-100:])
                        avg_steps = np.mean(episode_lengths[-100:])
                    else:
                        avg = np.mean(episode_rewards)
                        avg_steps = np.mean(episode_lengths[-100:])
                    avg_episode_rewards.append(avg)
                    avg_episode_lengths.append(avg_steps)
                    if args.verbose:
                        print(f"global step={global_step}, avg episode reward (100 eps)={avg:.2f},"
                              f" avg episode length (100 eps)={avg_steps:.2f}")
                    agent.reset()
                    
            state = next_state

        agent.update_policy(args.minibatch_size)
        agent.save_policy()
        np.save('episode_rewards_st_1', episode_rewards)
        np.save('avg_episode_rewards_st_1', avg_episode_rewards)
        np.save('episode_lengths_st_1', episode_lengths)
        np.save('avg_episode_lengths_st_1', avg_episode_lengths)

    if args.verbose:
        print(f"Saved {agent.get_name()}_{agent.policy_file_suffix}")

    if args.verbose:
        plt.ioff()
        plt.plot(episode_rewards)
        plt.plot(avg_episode_rewards)
        plt.legend(["Mean reward", "100-episode average"])
        plt.savefig("PPO_rewards_st_1.png")
        if args.visual:
            plt.show()

        plt.close()
        plt.ioff()
        plt.plot(episode_lengths)
        plt.plot(avg_episode_lengths)
        plt.legend(["Mean episode lengths", "100-episode average"])
        plt.savefig("PPO_ep_lengths_st_1.png")
        if args.visual:
            plt.show()

        plt.close()


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def train_simple_opponent(args):
    env_name = "WimblepongVisualSomeAI-v0"
    env = gym.make(env_name)
    #env = ParallelEnvs(env_name, processes=4, envs_per_process=1)
    env = SubprocVecEnv([make_env(env_name, args.seed + i)
                         for i in range(args.num_envs)], start_method="spawn")
    env = VecFrameStack(env, n_stack = 4)
    if args.algorithm.lower() == "dqn":
        agent = DQNagent.Agent(env_name, env.observation_space, env.action_space)
    elif args.algorithm.lower() == "ppo":
        agent = ppo_agent_st_1.Agent()
        agent.init_memory(args.steps_per_env, args.num_envs)
        agent.is_training = True
        if args.checkpoint:
            agent.load_checkpoint()
        elif args.pretrained_model:
            agent.load_model()
    else:
        raise NotImplementedError(f"No such algorithm: {args.algorithm.lower()}")

    train(env, agent, args)
    agent.save_policy()
    env.close()


def train_specific_opponent(args):
    env_name = "WimblepongMultiplayer-v0"
    env = gym.make(env_name)

    opponent = None
    if args.opponent_type.lower() == "dqn":
        opponent = DQNagent.Agent(env_name, env.observation_space, env.action_space)
        # opponent.load()
    elif args.opponent_type.lower() == "ppo":
        opponent = ppo_agent.Agent()
        # opponent.load()

    agent = None
    if args.algorithm.lower() == "dqn":
        agent = DQNagent.Agent(env_name, env.observation_space, env.action_space)
    elif args.algorithm.lower() == "ppo":
        agent = ppo_agent.Agent()

    train()


def main():
    parser = argparse.ArgumentParser(description="Train an agent in the Wimblepong environment")

    # Agent arguments
    parser.add_argument("--agent-name", type=str, default="mr_pong", help="Name of the agent")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Which learning algorithm to use")

    # Save paths
    parser.add_argument("--model-save-dir", type=str, default="./models/", help="Where to save the trained model")
    parser.add_argument("--plots-save-dir", type=str, default="./plots/", help="Where to save the plots")

    # Opponent arguments
    parser.add_argument("--opponent-type", type=str, default="simple",
                        help="Opponent type, defines the agent class. Options: simple [default], dqn, ppo")
    parser.add_argument("--opponent-file", type=str, default=None, help="File path of the opponent")

    # Hyperparameters
    parser.add_argument("--timesteps", type=int, default=2000000, help="Number of timesteps to train")
    parser.add_argument("--n_minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--steps-per-env", type=int, default=1000, help="Run each environment for N timesteps")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel game environments")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the experiment")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Print messages during training")
    parser.add_argument("--param-file", type=str, default="./src/agents/utils/ppo_params.txt",
                        help="Hyperparameter file path")
    parser.add_argument("--model-name", type=str, default="wimblepong_ppo", help="Name of the model")
    parser.add_argument("--model-save-path", type=str, default="./models", help="Where to save the trained model")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="if toggled, 'torch.backends.cudnn.deterministic=False'")
    parser.add_argument("--pretrained-model", action="store_true", default=False,
                        help="Toggle to load a pretrained model (for curriculum training).")
    parser.add_argument("--checkpoint", action="store_true", default=False,
                        help="Toggle to load a model checkpoint (for continuing a training session).")
    parser.add_argument("--survival-bonus", action="store_true", default=False,
                        help="Toggle to load give a survival bonus during training.")

    args = parser.parse_args()
    args.visual = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    args.batch_size = int(args.num_envs * args.steps_per_env)
    args.minibatch_size = int(args.batch_size // args.n_minibatches)

    if args.opponent_type.lower() == "simple":
        train_simple_opponent(args)
    else:
        train_specific_opponent(args)


if __name__ == "__main__":
    main()

