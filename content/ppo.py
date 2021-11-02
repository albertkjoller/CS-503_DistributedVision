from typing import Callable

import torch
from torch import optim
import torch.nn as nn
import torch.distributions as distributions


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class DataBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Policy:

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        """"""
        self.actor = actor
        self.critic = critic

    def act(self, state: torch.tensor) -> torch.tensor:
        """
        Uses the actor to sample an action to take in the given state.
        """
        policy_distribution = distributions.Categorical(self.actor(state))

        action = policy_distribution.sample()
        action_log_prob = policy_distribution.log_prob(action)

        return action.detach(), action_log_prob.detach()

    def state_value(self, state: torch.tensor) -> torch.tensor:
        """
        Uses the critic to evaluate the value of the state.
        """
        state_value = self.critic(state)
        return state_value.detach()


class PPO:

    def __init__(
        self,
        policy_generator: Callable[[], Policy],
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        optimization_epochs: int,
        epsilon_clip: float,
        device: str,
    ) -> None:
        """"""
        self.device = device
        self.policy_generator: Callable[[], Policy] = policy_generator

        self.policy: Policy = policy_generator()
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss: nn.Module = loss

        self.optimization_epochs = optimization_epochs
        self.epsilon_clip = epsilon_clip

    def clip(self, probability_ratios: torch.tensor) -> torch.tensor:
        """ clip(r_t, 1 - eps, 1 + eps) """
        return torch.clamp(probability_ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip)

    def act(self, state: torch.tensor) -> torch.tensor:
        """
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def set_reward(self, reward, is_terminal):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)

    def retrain(self, states, rewards):
        """
        """
        old_policy = self.policy_generator()
        old_policy.load_state_dict(self.policy.state_dict())

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = self.clip(ratios) * advantages

            # final loss of clipped objective PPO
            loss = torch.min(surr1, surr2)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()
