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
from collections import defaultdict
from multiprocessing import cpu_count, Process
import matplotlib.pyplot as plt


class Environment:
    """
    Environment setup - using VizDoom variable names for the game environment to create an OpenAI-like instance
    for consistency with OpenAI gym tools
    """

    def __init__(self, config_file_path, players, window_visible=False, depth=False, skip=4):

        self.config_file_path = config_file_path
        self.window_visible = window_visible
        self.depth = depth
        self.players = players
        self.skip = 4

        self.initialize_player()
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.actions_size = len(self.actions)

    def initialize_player(self):
        """
        Load configuration from path...
        """
        print("")
        self.game = vizdoom.DoomGame()
        self.game.load_config(self.config_file_path)
        self.game.set_window_visible(self.window_visible)
        self.game.set_mode(vizdoom.Mode.PLAYER) #TODO: CHECK MODE Async or not
        self.game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        self.game.set_depth_buffer_enabled(self.depth)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        print("Player created!")

    def step(self, a):
        """
        Step function for interaction in the environment
        """
        r = self.game.make_action(a)
        done = self.game.is_episode_finished()
        sp = self.game.get_state().screen_buffer if not done else 0 # TODO: could be done smarter, state could include depth
        return sp, r, done


class Agent:
    """ Main agent class. Three items: 1) initialization, 2) policy and 3) training step"""
    def __init__(self, env):
        self.env = env
        self.states = defaultdict(dict)
        self.rewards = {}
        self.actions_players = {}

        self.time_step = 0

    def pi(self, s, k=None):
        """
        Policy of the agent
        """
        # Random policy
        # TODO: trying to append states to train central agent on multiple screen views - not working right now
        k = self.time_step
        if k in self.states.keys():
            self.states[k].append(s)
        else:
            self.states[k] = [s]

        random_idx = np.random.choice(self.env.actions_size)
        a = self.env.actions[random_idx]
        return a

    def train(self, s, a, r, sp, done=False):
        """
        Algorithm for training the agent
        """
        # Currently no training
        pass


def player_join(p):
    #Manual setup - would be nice to find a solution in an object-related form
    config_file_path = "../setting/settings.cfg"
    cameras = 4  # number of players
    num_episodes = 100
    episode_length = 200  # 2100
    env = Environment(config_file_path=config_file_path, window_visible=True, depth=True, players=cameras)
    agent = Agent(env)

    env.initialize_player()
    env.game.add_game_args("-join 127.0.0.1 +name Player" + str(p) + " +colorset " + str(p))

    env.game.init()

    for i in range(num_episodes):
        a = 1
        env.inti

        while not env.game.is_episode_finished():
            s = env.game.get_state()
            # screen = s.screen_buffer
            a = agent.pi(s)
            env.step(a)

        env.game.new_episode()

    env.game.close()

def train(env, agent, episodes, episode_length=2100):
    """General training loop - applicable to multiple different agents"""

    processes = []
    for i in range(1, env.players):
        p_join = Process(target=player_join, args=(i,))
        p_join.start()
        processes.append(p_join)

        env.initialize_player()
        env.game.add_game_args("-join 127.0.0.1 +name Player" + str(i) + " +colorset " + str(i))


    env.initialize_player()
    env.game.add_game_args("-host " + str(env.players) + " -netmode 0 +timelimit " + str(episode_length) +
                       " +sv_spawnfarthest 1 +name Player0 +colorset 0")

    env.game.set_episode_timeout(episode_length)
    env.game.init()

    for episode_iteration in tqdm(range(episodes)):
        env.game.new_episode()

        agent.time_step = 0
        reward = []

        with tqdm(total=env.game.get_episode_timeout(), desc=f"Episode {episode_iteration}", position=0, leave=True, colour='green') as tq:
            while not env.game.is_episode_finished():

                s = env.game.get_state()

                a = agent.pi(s=s)
                sp, r, done = env.step(a)

                agent.train(s, a, r, sp, done)

                agent.time_step += 1
                reward.append(r)
                tq.update()

    env.game.close()




if __name__ == '__main__':

    config_file_path = "../setting/settings.cfg"
    cameras = 4  # number of players
    num_episodes = 100
    episode_length = 200  # 2100

    env = Environment(config_file_path=config_file_path, window_visible=True, depth=True, players=cameras)
    agent = Agent(env)

    train(env, agent, episodes=num_episodes, episode_length=episode_length)
