"""
CS-503 - Distributed Vision (Burmester Felix, Jacobsen Albert, Poulsen Niels)

Basic script for the setup of the Reinforcement Learning problem. Contains three major instances:
1) the 'Environment' class, specifying the VizDoom game setup and the step-function for updating env. when interacting
2) the 'Agent' class, specifying the policy of an agent as well as the algorithm used for training
3) the training loop, simulating an agents interaction with the environment and updating its learning parameters
"""

from __future__ import print_function

from tqdm import tqdm
from vizdoom import vizdoom
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torch.distributions as distributions
from torchvision.models import resnet18
from torchvision import transforms


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # actor
        self.actor = resnet18(pretrained=False)
        self.actor.fc = nn.Sequential(
            nn.Linear(
                self.actor.fc.in_features,
                action_dim
            ),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = resnet18(pretrained=False)
        self.critic.fc = nn.Linear(
            self.critic.fc.in_features,
            1
        )

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        norm_state = self.normalize(state)
        action_probs = self.actor(norm_state)
        dist = distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        norm_state = self.normalize(state)
        action_probs = self.actor(norm_state)
        dist = distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(norm_state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).permute((2, 0, 1)).unsqueeze(0).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


class Environment():
    """
    Environment setup - using VizDoom variable names for the game environment to create an OpenAI-like instance
    for consistency with OpenAI gym tools
    """

    def __init__(self, config_file_path, window_visible=False, depth=False, labels=False):
        move_buttons = [
            vizdoom.Button.MOVE_LEFT,
            vizdoom.Button.MOVE_RIGHT,
            vizdoom.Button.MOVE_FORWARD,
            vizdoom.Button.MOVE_BACKWARD,
        ]
        turn_buttons = [
            vizdoom.Button.TURN_LEFT,
            vizdoom.Button.TURN_RIGHT
        ]
        buttons = move_buttons + turn_buttons

        self.initialize_vizdoom(config_file_path, buttons, window_visible, depth, labels)
        n = self.game.get_available_buttons_size()
        self.actions = [n * [0] for _ in range(n)]
        for i in range(n):
            self.actions[i][i] = 1
        self.actions_size = len(self.actions)

    def initialize_vizdoom(self, config_file_path, buttons, window_visible=False, depth=False, labels=False):
        """
        Load configuration from path...
        """

        print("Initializing doom...")
        self.game = vizdoom.DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(False)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        #self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        self.game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        self.game.set_depth_buffer_enabled(depth)
        self.game.set_labels_buffer_enabled(labels)
        self.game.set_available_buttons(buttons)

        print("Doom setup succesfull.")

    def step(self, action: int):
        """
        Step function for interaction in the environment
        """
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        state = self.game.get_state().screen_buffer if not done else None
        return state, reward, done


def train(env, agent, episodes, max_ep_len, update_timestep):
    """General training loop - applicable to multiple different agents"""

    env.game.init()
    time_step = 0
    for episode_iteration in tqdm(range(episodes)):
        env.game.new_episode()
        state = env.game.get_state().screen_buffer
        episode_reward = 0
        for _ in tqdm(range(max_ep_len), desc=f"Episode {episode_iteration}", position=0, leave=True, colour='green'):
            # select action with policy
            action = agent.select_action(state)
            state, reward, done = env.step(action)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            episode_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()

            if done:
                break


if __name__ == '__main__':

    config_file_path = "setting/settings.cfg"
    
    state_dim = (640, 480)  # RES_640X480
    action_dim = 6          # [forward, back, left, right, move left, move right]

    ################ PPO hyperparameters ################
    max_ep_len = 100                      # max timesteps in one episode
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80                         # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    env = Environment(config_file_path=config_file_path, window_visible=True, depth=True, labels=True)
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip
    )

    num_episodes = 100
    train(env, ppo_agent, episodes=num_episodes, max_ep_len=max_ep_len, update_timestep=update_timestep)
