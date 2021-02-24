import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time


class CNN(torch.nn.Module):
    def __init__(self, state_space, n_actions, device=torch.device("cpu")):
        super(CNN, self).__init__()
        self.device = device
        self.hidden_neurons = 512
        self.stack = 4
        channels = self.stack
        self.n_actions = n_actions

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, kernel_size=5, stride=3, padding=0),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU()
        )
        
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(5 * 5 * 64, self.hidden_neurons),
            torch.nn.Tanh()
        )
       
        self.action_layer = torch.nn.Linear(self.hidden_neurons, self.n_actions)
        self.value_layer = torch.nn.Linear(self.hidden_neurons, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)


    def evaluate(self, state, action):
        action_dist, state_value = self.forward(state)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return log_prob, entropy

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        
        
        action_mean = self.action_layer(x)
        action_dist = torch.distributions.Categorical(F.log_softmax(action_mean, dim=-1))
        state_value = self.value_layer(x)
        return action_dist, state_value.squeeze(-1)

class Agent(object):
    """
    Agent using a minimal actor-critic neural network and Proximal Policy Optimization
    """
    def __init__(self):
        self.name = "pong_roger"
        self.policy_file_suffix = "ppo_policy_5.pth"
        hyperparam_file = "src/agents/utils/ppo_params.txt"
        with open(hyperparam_file) as file:
            lines = file.readlines()
            if lines[0].strip() != "PPO hyperparameters":
                raise ValueError("Incorrect file identifier")
            lines = lines[1:]
            params = {line.split("=")[0].strip(): line.split("=")[1].strip() for line in lines}

        # Set to true in training script to return actions as tensors, leave to false in
        # testing scripts to return integer (compliant with the project interface)
        self.is_training = False

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
        self.state_space = (200, 200, 3)
        self.action_space = 3
        self.policy = CNN(self.state_space, self.action_space, torch.device(self.training_device))
        self.policy = self.policy.to(self.training_device)
        self.stack = 4

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.MseLoss = torch.nn.MSELoss()
        
        # State transition buffers
        self.states = None
        self.state_values = None
        self.actions = None
        self.action_probs = None
        self.rewards = None
        self.dones = None

        self.prev_obs = []

    def init_memory(self, num_steps, num_envs): 
        downsampled_dims = (self.stack, self.state_space[0]//4, self.state_space[1]//4)
        self.states = torch.zeros((num_steps, num_envs) + downsampled_dims).to(self.training_device)
        self.state_values = torch.zeros((num_steps, num_envs)).to(self.training_device)
        self.actions = torch.zeros((num_steps, num_envs)).to(self.training_device)
        self.action_probs = torch.zeros((num_steps, num_envs)).to(self.training_device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(self.training_device)
        self.dones = torch.zeros((num_steps, num_envs)).to(self.training_device)
        

    def get_action(self, observation):
        """
        Given the observation, stochastically choose an action following the old policy.
        :param observation: observed state, has the shape of the environment state space vector.
        :return: chosen action, logarithmic probability of the action, and distribution entropy
        """
        
        if len(observation.shape)==3:
            state = self.test_preprocess(observation)
        else:
            state = self.preprocess(observation)
        
        action_dist, _ = self.policy.forward(state)

        action = action_dist.sample()
        if self.is_training:
            return action

        return torch.argmax(action_dist.probs).item()

    def evaluate(self, state, action):
        if type(state) is torch.Tensor:
            obs = state
        else:
            obs = self.preprocess(state)
        return self.policy.evaluate(obs, action)

    def update_policy(self, minibatch_size):
        """
        Update the policy with PPO. Gets the necessary data from state transition buffers.
        :param minibatch_size: size of the minibatch for optimization
        :return: Nothing
        """
       
        steps = self.rewards.shape[0]
        batch_size = self.rewards.shape[0] * self.rewards.shape[1]
        returns  = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        
        if self.gae:
            for t in reversed(range(steps)):
                if t == steps-1:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.state_values[t]
                    td_error = returns[t] - self.state_values[t]
                else:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * returns[t+1]
                    td_error = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.state_values[t+1] - self.state_values[t]
                advantages[t] = advantages[t] * self.gae_lambda * self.gamma * (1-self.dones[t]) + td_error
            
        else:
            for t in reversed(range(steps)):
                if t == steps-1:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.state_values[t]
                else:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * returns[t+1]
                advantages[t] = returns[t] - self.state_values[t]

        b_obs = self.states.reshape((-1,) + (self.stack,50,50))
        b_logprobs = self.action_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.state_values.reshape(-1)
        

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

                newlogproba, entropy = self.evaluate(b_obs[minibatch_ind], b_actions[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                _, new_values = self.policy.forward(b_obs[minibatch_ind])
                if self.clip_vloss:           
                    v_loss_unclipped = self.MseLoss(new_values,b_returns[minibatch_ind])
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind],
                                                                      -self.clip_epsilon, self.clip_epsilon)
                    v_loss_clipped = self.MseLoss(v_clipped,b_returns[minibatch_ind])
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * self.MseLoss(new_values,b_returns[minibatch_ind])

                loss = pg_loss + v_loss * self.vf_coeff - self.ent_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def preprocess(self, obs):
        '''
        if len(obs.shape)==3: #when testing, observation shape is just (200,200,3)
            obs = np.expand_dims(obs,0) #(1, 200, 200, 3)
        obs = obs/255.0
        obs = obs[::, ::2, ::2].mean(axis=-1)
        obs = np.expand_dims(obs, 1)
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float().to(self.training_device)
        return obs
        '''
        obs = obs/255.0
        single_channel_imgs = []
        for n in range(obs.shape[-1]//3):
            mean_img = obs[:,::4,::4, n*3:n*3+3].mean(axis=-1)
            mean_img = np.expand_dims(mean_img, 1)
            single_channel_imgs.append(torch.from_numpy(mean_img).float().to(self.training_device))
        obs = torch.cat(single_channel_imgs, dim=1)
        return obs
    
    def test_preprocess(self, obs):
        #obs = (200x200x3)
        obs = np.expand_dims(obs,0) #(1, 200, 200, 3)
        self.prev_obs.append(obs)
        if len(self.prev_obs)==5:
            self.prev_obs.pop(0)
            
        if len(self.prev_obs)==1:
            obs = np.concatenate([obs,obs,obs,obs],axis=-1)
        elif len(self.prev_obs)==2:
            obs = np.concatenate([self.prev_obs[0],self.prev_obs[0],self.prev_obs[0],self.prev_obs[1]],axis=-1)
        elif len(self.prev_obs)==3:
            obs = np.concatenate([self.prev_obs[0],self.prev_obs[0],self.prev_obs[1],self.prev_obs[2]],axis=-1) 
        elif len(self.prev_obs)==4:
            obs = np.concatenate(self.prev_obs[-4:],axis=-1)
        
        obs = obs/255.0
        single_channel_imgs = []
        for n in range(obs.shape[-1]//3):
            mean_img = obs[:,::4,::4, n*3:n*3+3].mean(axis=-1)
            mean_img = np.expand_dims(mean_img, 1)
            single_channel_imgs.append(torch.from_numpy(mean_img).float().to(self.training_device))
        obs = torch.cat(single_channel_imgs, dim=1)
        return obs
        
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
        p_state = self.preprocess(state)
        self.states[step] = p_state.to(self.training_device)
        self.actions[step] = action.to(self.training_device)
        self.action_probs[step] = action_prob.to(self.training_device)
        self.rewards[step] = torch.from_numpy(np.clip(reward, -1, 1)).float().to(self.training_device)
        with torch.no_grad():
            _, state_values = self.policy.forward(p_state)
            self.state_values[step] = state_values
        self.dones[step] = torch.Tensor([done]).to(self.training_device)

    def save_policy(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f"./{self.name}_{self.policy_file_suffix}")

    def load_model(self):
        model = torch.load(f"./{self.name}_{self.policy_file_suffix}",map_location=torch.device('cpu'))
        self.policy.load_state_dict(model['policy_state_dict'])

    def load_checkpoint(self):
        model = torch.load(f"./{self.name}_{self.policy_file_suffix}")
        self.policy.load_state_dict(model['policy_state_dict'])
        self.optimizer.load_state_dict(model['optimizer_state_dict'])

    def get_name(self):
        return self.name
    
    def reset(self):
        self.prev_obs = []   
        
