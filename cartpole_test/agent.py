import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 256
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        #self.sigma = torch.zeros(1)  # TODO: Implement learned variance (or copy from Ex5)
        self.sigma = torch.nn.Parameter(torch.tensor([10.0], requires_grad = True))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def evaluate(self, state, action):
        action_dist, state_value = self.forward(state)
        #print(action_dist)
        #print(action_dist.mean.shape)
        log_prob = action_dist.log_prob(action.unsqueeze(1))
        #log_prob = action_dist.log_prob(action)
        print(log_prob)
        entropy = action_dist.entropy()
        return log_prob, entropy

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)
        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = F.softplus(self.sigma)
        value = self.fc3(x)     
        action_dist = Normal(action_mean, sigma)
        return action_dist,value
    
class Agent(object):
    """
    Agent using a minimal actor-critic neural network and Proximal Policy Optimization
    """
    def __init__(self):
        self.name = "pong_bot"
        self.policy_file_suffix = "ppo_policy.pth"
        hyperparam_file = "ppo_params.txt"
        with open(hyperparam_file) as file:
            lines = file.readlines()
            if lines[0].strip() != "PPO hyperparameters":
                raise ValueError("Incorrect file identifier")
            lines = lines[1:]
            params = {line.split("=")[0].strip(): line.split("=")[1].strip() for line in lines}

        self.learning_rate = float(params["learning_rate"])
        self.gamma = float(params["gamma"])
        self.epochs = int(params["epochs"])
        self.clip_epsilon = float(params["clip_epsilon"])
        self.vf_coeff = float(params["vf_coeff"])
        self.ent_coeff = float(params["ent_coeff"])
        self.norm_adv = params["norm_adv"].lower() == "true"
        self.clip_vloss = params["clip_vloss"].lower() == "true"
        self.max_grad_norm = float(params["max_grad_norm"])
        self.gae = params["gae"].lower() == "true"
        self.gae_lambda = float(params["gae_lambda"])

        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up the policy
        self.state_space = 4
        self.action_space = 1
        self.policy = Policy(self.state_space, self.action_space)
        self.policy = self.policy.to(self.training_device)
        self.old_policy = Policy(self.state_space, self.action_space).to(self.training_device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.MseLoss = torch.nn.MSELoss()

        # State transition buffers
        self.states = None
        self.state_values = None
        self.actions = None
        self.action_probs = None
        self.rewards = None
        self.dones = None

    def init_memory(self, num_steps):
        self.states = torch.zeros((num_steps, self.state_space )).to(self.training_device)
        self.state_values = torch.zeros((num_steps, 1)).to(self.training_device)
        self.actions = torch.zeros((num_steps, 1)).to(self.training_device)
        self.action_probs = torch.zeros((num_steps, 1)).to(self.training_device)
        self.rewards = torch.zeros((num_steps, 1)).to(self.training_device)
        self.dones = torch.zeros((num_steps, 1)).to(self.training_device)
        
    def get_action(self, state):
        """
        Given the observation, stochastically choose an action following the old policy.
        :param observation: observed state, has the shape of the environment state space vector.
        :return: chosen action, logarithmic probability of the action, and distribution entropy
        """
        state = torch.from_numpy(state).float().to(self.training_device)
        action_dist, _ = self.old_policy.forward(state)
        action = action_dist.sample()

        return action

    def evaluate(self, state, action):
         if type(state) is torch.Tensor:
             #print('new')
             obs = state
             return self.policy.evaluate(obs, action)
         else:
             #print('old')
             obs = torch.from_numpy(state).float().to(self.training_device)
             return self.old_policy.evaluate(obs, action)
    
    def update_policy(self, minibatch_size):
        """
        Update the policy with PPO. Gets the necessary data from state transition buffers.
        :param minibatch_size: size of the minibatch for optimization
        :return: Nothing
        """
        
        steps = self.rewards.shape[0]
        batch_size = self.rewards.shape[0] * self.rewards.shape[1]
        #steps = 500
        #batch_size = 500
        #print(steps)
        #print(batch_size)
        
        # Compute advantages
        '''
        with torch.no_grad():
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.training_device)
                lastgaelam = 0
                for t in reversed(range(steps)):
                    if t == steps - 1:
                        nextnonterminal = 1.0 - self.dones[t]
                        nextvalues = self.state_values[t]
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.state_values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.state_values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.state_values
            else:
                returns = torch.zeros_like(self.rewards).to(self.training_device)
                for t in reversed(range(steps)):
                    if t == steps - 1:
                        nextnonterminal = 1.0 - self.dones[t]
                        next_return = self.state_values[t]
                    else:
                        nextnonterminal = 1.0 - self.dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.state_values
        '''        
        returns = torch.zeros_like(self.rewards).to(self.training_device)
        for t in reversed(range(steps)):
            if t == steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                next_return = self.state_values[t]
            else:
                nextnonterminal = 1.0 - self.dones[t+1]
                next_return = returns[t+1]
            returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
        advantages = returns - self.state_values
          

        # flatten the batch
        #b_obs = self.states.reshape((-1,) + self.state_space)
        #print(self.states.shape)
        b_obs = self.states.reshape((-1,4)).detach()
        b_logprobs = self.action_probs.reshape(-1,1).detach()
        b_actions = self.actions.reshape((-1,)).detach()
        b_advantages = advantages.reshape(-1,1)
        b_returns = returns.reshape(-1,1)
        b_values = self.state_values.reshape(-1,1)
        
        # Optimize policy and value network for K epochs, run optimization in minibatches
        
        inds = np.arange(batch_size)
        for i_epoch_pi in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                   
                #_, newlogproba, entropy = self.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])
                newlogproba, entropy = self.evaluate(b_obs[minibatch_ind], b_actions[minibatch_ind])
                #ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()
                ratio = torch.exp((newlogproba - b_logprobs[minibatch_ind].detach()))
        
                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                _, new_values = self.policy.forward(b_obs[minibatch_ind])
                if self.clip_vloss:
        
                    v_loss_unclipped = self.MseLoss(new_values,b_returns[minibatch_ind])
                    #v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind],
                                                                      -self.clip_epsilon, self.clip_epsilon)
                    #v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_clipped = self.MseLoss(v_clipped,b_returns[minibatch_ind])
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    #v_loss = 0.5 * v_loss_max.mean()
                    v_loss = 0.5 * v_loss_max
                else:
                    #v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()
                    v_loss = self.MseLoss(new_values,b_returns[minibatch_ind])

                loss = pg_loss + v_loss * self.vf_coeff - self.ent_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # Copy new weights into old policy:
        self.old_policy.load_state_dict(self.policy.state_dict())
     
        
        
    def update_(self):
        steps = self.rewards.shape[0]
        batch_size = self.rewards.shape[0] * self.rewards.shape[1]
        
        #compute advantage
        returns = torch.zeros_like(self.rewards).to(self.training_device)
        for t in reversed(range(steps)):
            if t == steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                next_return = self.state_values[t]
            else:
                nextnonterminal = 1.0 - self.dones[t+1]
                next_return = returns[t+1]
            returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = returns - self.state_values
        
        b_obs = self.states.reshape((-1,4)).detach()
        b_logprobs = self.action_probs.reshape(-1,1).detach()
        b_actions = self.actions.reshape((-1,)).detach()
        b_advantages = advantages.reshape(-1,1)
        b_returns = returns.reshape(-1,1)
        b_values = self.state_values.reshape(-1,1)
        
        for i_epoch_pi in range(self.epochs):
            #mb_advantages = b_advantages
            newlogproba, entropy = self.evaluate(b_obs, b_actions)
            _,state_v = self.policy(b_obs)
            state_v = state_v.reshape(-1,1)
            e_advantages = returns - state_v.detach()
            ratio = torch.exp((newlogproba - b_logprobs.detach()))
            surr1 = ratio * e_advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * e_advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_v, b_returns) - 0.01*entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.old_policy.load_state_dict(self.policy.state_dict())
            
        
        
    def store_outcome(self, step, state, action, action_prob, reward, done):
        """
        Store the outcome of a timestep into the state transition buffers.
        :param step: the current timestep, i.e. index in memory
        :param state: the state where the action was taken
        :param action: the action that was taken
        :param action_prob: logarithmic probability of the action
        :param reward: immediate reward
        :param done: true if next_state is terminal
        :return: Nothing
        """
        state = torch.from_numpy(state).float()
        self.states[step] = state.to(self.training_device)
        self.actions[step] = action.to(self.training_device)
        self.action_probs[step] = action_prob.to(self.training_device)
        self.rewards[step] = torch.from_numpy(np.asarray(np.clip(reward, -1, 1))).float().to(self.training_device)
        with torch.no_grad():
            _, state_values = self.policy.forward(state)
            self.state_values[step] = state_values
        self.dones[step] = torch.Tensor([done]).to(self.training_device)

