from wimblepong import Wimblepong
import random
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

class SomeAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4              
        self.name = "Some_AI"
        
        self.state_space = (200, 200, 3)
        self.action_space = 3
        self.policy = CNN(self.state_space, self.action_space)
        self.prev_obs = []
        self.training_device = torch.device("cpu")
        self.is_training = True
        self.load_model()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self,observation = None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        observation = self.env._get_observation(self.player_id)
        if len(observation) != 0:
             
            if len(observation.shape)==3:
                state = self.test_preprocess(observation)
            else:
                state = self.preprocess(observation)
        
            action_dist, _ = self.policy.forward(state)

            action = action_dist.sample()
            if self.is_training:
                return action

            return torch.argmax(action_dist.probs).item()
        elif observation == None:
            # Get the player id from the environmen
            player = self.env.player1 if self.player_id == 1 else self.env.player2
            # Get own position in the game arena
            my_y = player.y
            # Get the ball position in the game arena
            ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)

            # Compute the difference in position and try to minimize it
            y_diff = my_y - ball_y
            if abs(y_diff) < 2:
                action = 0  # Stay
            else:
                if y_diff > 0:
                    action = self.env.MOVE_UP  # Up
                else:
                    action = self.env.MOVE_DOWN  # Down
            #action = self.env.STAY
            return action
        

    def reset(self):
        # Nothing to done for now...
        self.prev_obs = []
        #return
    
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
    
    def load_model(self):
        model = torch.load('some_ai.pth',map_location=torch.device('cpu'))
        self.policy.load_state_dict(model['policy_state_dict'])

    
        
