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


class Environment():
    """
    Environment setup - using VizDoom variable names for the game environment to create an OpenAI-like instance
    for consistency with OpenAI gym tools
    """

    def __init__(self, config_file_path, window_visible=False):

        self.initialize_vizdoom(config_file_path, window_visible)
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.actions_size = len(self.actions)

    def initialize_vizdoom(self, config_file_path, window_visible=False):
        """
        Load configuration from path...
        """

        print("Initializing doom...")
        self.game = vizdoom.DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(window_visible)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        print("Doom setup succesfull.")

    def step(self, a):
        """
        Step function for interaction in the environment
        """

        r = self.game.make_action(a)
        done = self.game.is_episode_finished()

        # TODO: Make it more smooth - in case we want to use depth and label buffers.
        # TODO: Also update of state when done could be done smarter
        sp = self.game.get_state().screen_buffer if not done else 0
        return sp, r, done


class Agent:
    """ Main agent class. Three items: 1) initialization, 2) policy and 3) training step"""
    def __init__(self, env):
        self.env = env

    def pi(self, s, k=None):
        """
        Policy of the agent
        """
        # Random policy
        random_idx = np.random.choice(self.env.actions_size)
        return self.env.actions[random_idx]

    def train(self, s, a, r, sp, done=False):
        """
        Algorithm for training the agent
        """
        # Currently no training
        pass


def train(env, agent, episodes):
    """General training loop - applicable to multiple different agents"""

    env.game.init()
    for episode_iteration in tqdm(range(episodes)):
        env.game.new_episode()

        time_step = 0
        reward = []

        with tqdm(total=env.game.get_episode_timeout(), desc=f"Episode {episode_iteration}", position=0, leave=True, colour='green') as tq:
            while not env.game.is_episode_finished():
                # TODO: Determine notation of states, actions, etc. (OpenAI gym or VizDoom examples?)
                # TODO: As well as which features to use
                s = env.game.get_state()
                screen = s.screen_buffer
                depth = s.depth_buffer
                labels = s.labels_buffer
                automap = s.automap_buffer
                labels = s.labels

                a = agent.pi(s=screen)
                sp, r, done = env.step(a)
                agent.train(s, a, r, sp, done)

                time_step += 1
                reward.append(r)
                tq.update()


            #TODO: Depending on the policy, we might want to save the trajectories?



if __name__ == '__main__':

    config_file_path = "setting/settings.cfg"

    env = Environment(config_file_path=config_file_path, window_visible=True)
    agent = Agent(env)

    num_episodes = 10
    train(env, agent, episodes=num_episodes)

