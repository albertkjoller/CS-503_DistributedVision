"""
Functions are re-written but taken mainly from the ViZDoom/examples/python/multiple_instances_advance.py file
"""

import os
from random import choice, random
from time import sleep, time
import vizdoom


class Players:
    def __init__(self, config_file_path, window_visible=False, depth=False):
        self.config_file_path = config_file_path
        self.window_visible = window_visible
        self.depth = depth

    def setup_player(self):
        """
        Load configuration from path...
        """
        print("Initializing doom...")
        self.game = vizdoom.DoomGame()
        self.game.load_config(self.config_file_path)
        self.game.set_window_visible(self.window_visible)
        self.game.set_mode(vizdoom.Mode.PLAYER) #TODO: CHECK MODE Async or not
        self.game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        self.game.set_depth_buffer_enabled(self.depth)
        print("Doom setup succesfull.")

    def player_host(self, p):
        game, actions = self.setup_player()
        game.add_game_args("-host " + str(p) + " -netmode 0 -deathmatch +timelimit " + str(timelimit) +
                           " +sv_spawnfarthest 1 +name Player0 +colorset 0")
        game.init()

        game.close()

    def player_join(self, p):
        game, actions = self.setup_player()
        game.add_game_args("-join 127.0.0.1 +name Player" + str(p) + " +colorset " + str(p))
        #game.add_game_args(args)

        game.init()

        for i in range(episodes):

            while not game.is_episode_finished():
                state = game.get_state()
                print("Player" + str(p) + ":", state.number, action_count, game.get_episode_time())
                player_action(game, player_sleep_time, actions, player_skip)
                action_count += 1

            print("Player" + str(p) + " frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            game.new_episode()

        game.close()

