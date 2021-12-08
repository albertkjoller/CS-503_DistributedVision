import argparse

from os import read
from pathlib import Path
from time import sleep
from typing import List, Tuple, Callable
from threading import Thread

import cv2 as cv2
import gym as gym
import matplotlib.pyplot as plt
import numpy as np

from vizdoom import vizdoom
from gym.spaces import Space, Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common import callbacks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


class SingleAgentEnv(gym.Env):
    """ Environment used for the baseline """

    def __init__(
        self,
        game: vizdoom.DoomGame,
        frame_processor: Callable,
        frame_skip: int,
        save_occupancy_maps: bool,
    ):
        super().__init__()
        print("Initializing DistributedVisionEnvironment")
        self.game = game
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip
        self.save_occupancy_maps = save_occupancy_maps

        self.map_x_boundaries = 160.0, 1120.0  # (min_x, max_x)
        self.map_y_boundaries = -704.0, 128.0  # (min_y, max_y)
        self.map_shape = (
            self.map_x_boundaries[1] - self.map_x_boundaries[0],
            self.map_y_boundaries[1] - self.map_y_boundaries[0]
        )
        self.occupancy_map_shape = (36, 24)  # (width, height) (36, 24)
        self.reset_occupancy_map()

        # The actions that can be taken
        self._game_button_size = game.get_available_buttons_size()

        # All combinations of actions possible
        self.action_space: Space = Discrete(self._game_button_size)
        self.possible_actions: list[list[int]] = np.eye(self.action_space.n).tolist()

        print(f"  * Action Space Size: {self._game_button_size}")
        print(f"  * Possible Actions:  {len(self.possible_actions)}")
        for action in self.possible_actions:
            print(f"    - {action}")

        out_h, out_w = None, None
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        out_h, out_w, out_channels = frame_processor(np.zeros((h, w, c))).shape
        out_channels += 1  # Occupancy map

        self.observation_space: Space = Box(
            low=0,
            high=255,
            shape=(out_h, out_w, out_channels),
            dtype=np.uint8
        )

        # Empty frame for when the game is done
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # The current state of the game 
        self.state = self.empty_frame

    def reset_occupancy_map(self):
        """"""
        self.occupancy_map = np.zeros(self.occupancy_map_shape, dtype=np.uint8)

    def update_occupancy_map(self) -> np.ndarray:
        pos_x = self.game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(vizdoom.GameVariable.POSITION_Y)

        grid_x = min(
            int(self.occupancy_map_shape[0] * ((pos_x - self.map_x_boundaries[0])/self.map_shape[0])),
            self.occupancy_map_shape[0] - 1
        )
        grid_y = min(
            int(self.occupancy_map_shape[1] * ((pos_y - self.map_y_boundaries[0])/self.map_shape[1])),
            self.occupancy_map_shape[1] - 1
        )

        self.occupancy_map[grid_x, grid_y] = 255  # Max of np.unint8

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        agent_action = self.possible_actions[action]
        rewards = 0
        done = False
        for _ in range(self.frame_skip):
            reward = self.game.make_action(agent_action, 1)
            rewards += reward
            self.update_occupancy_map()
            done = self.game.is_episode_finished()
            if done:
                break

        self.state = self._get_frame(done)
        return self.state, reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        if self.save_occupancy_maps:
            plt.figure()
            plt.title("Occupancy map")
            plt.imshow(self.occupancy_map)
            plt.savefig(f"../results/occ_map_episode_{self.epsiode_number}.jpg")
            plt.close()

        self.reset_occupancy_map()
        self.game.new_episode()
        self.state = self._get_frame()
        return self.state

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> np.ndarray:
        if done:
            return self.empty_frame

        actor_state = self.game.get_state()
        if actor_state is None or actor_state.screen_buffer is None:
            print("Found none state: had to return empty frame")
            return self.empty_frame
        
        game_state = [self.frame_processor(actor_state.screen_buffer)]
        game_state.append(
            np.expand_dims(self.frame_processor(self.occupancy_map), 2)
        )
        
        return np.concatenate(
            game_state,
            axis=2
        )


def create_env(
    config_file_path: Path = Path("../setting/settings.cfg"),
    game_port: str = "5029",
    player_id: int = 0,  # Player 0 is host
    num_players: int = 3,
    max_episode_length: int = 500,
    frame_skip: int = 1,
    screen_resolution: vizdoom.ScreenResolution = vizdoom.ScreenResolution.RES_320X240,
    frame_processor: Callable = lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    window_visible: bool = True,
    save_occupancy_maps: bool = False,
    buttons: List[vizdoom.Button] = [vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT, vizdoom.Button.ATTACK],
):
    game = vizdoom.DoomGame()
    game.load_config(str(config_file_path))
    game.set_window_visible(window_visible)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.set_episode_timeout(max_episode_length)

    if player_id == 0:
       game_args = (
            f"-host {num_players}" +
            f" -port {game_port}" +
            " -netmode 0" + 
            f" +timelimit {max_episode_length}" +
            " +sv_spawnfarthest 1" +
            " +name Player0" +
            " +colorset 0"
        )
    else:
        game_args = (
            "-join 127.0.0.1" +
            f" -port {game_port}" +
            f" +name Player{player_id}" +
            f" +colorset {player_id}"
        )

    print(f"Creating with game args: {game_args}")
    game.add_game_args(game_args)

    game.init()
    print(f"Creating player {player_id} for {num_players}-player game on port {game_port}")
    return SingleAgentEnv(game, frame_processor, frame_skip, save_occupancy_maps)


def create_vec_env(eval: bool = False, **kwargs) -> VecTransposeImage:
    """"""
    vec_env = VecTransposeImage(DummyVecEnv([lambda: create_env(**kwargs)]))
    
    if eval:
        vec_env = Monitor(vec_env)

    return vec_env


def create_agent(env, **kwargs):
    """"""
    return PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=4096,
        batch_size=32,
        learning_rate=1e-4,
        tensorboard_log='logs/tensorboard',
        verbose=0,
        seed=0,
        **kwargs
    )


def run_single_agent(player_id, config_train, config_eval, total_timesteps, eval_freq, n_eval_episodes):
    """"""
    player_config_train = config_train.copy()
    player_config_eval = config_eval.copy()

    player_config_train["player_id"] = player_id
    player_config_eval["player_id"] = player_id

    # Create training and evaluation environments.
    training_env = create_vec_env(**player_config_train)
    eval_env = create_vec_env(eval=False, **player_config_eval)

    # Create the agent
    agent = create_agent(training_env)

    # Define an evaluation callback that will save the model when a new reward record is reached.
    evaluation_callback = callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        log_path=f'logs/evaluations/ppo_multi_agent_baseline_player{player_id}',
        best_model_save_path=f'logs/models/ppo_multi_agent_baseline_player{player_id}'
    )

    # Play!
    agent.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f'ppo_multi_agent_baseline_player{player_id}',
        callback=evaluation_callback
    )

    # To view logs, run in another directory:
    #   tensorboard --logdir logs/tensorboard
    # And go to http://localhost:6006/ in Firefox or Chrome

    training_env.close()
    eval_env.close()



def run(
    save_occupancy_maps: bool,
    window_visible: bool,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
):

    # Configuration parameters
    base_config = {
        "config_file_path": Path("../setting/settings.cfg"),
        "screen_resolution": vizdoom.ScreenResolution.RES_320X240,
        "window_visible": window_visible,
        "buttons": [
            vizdoom.Button.MOVE_FORWARD,
            vizdoom.Button.TURN_LEFT,
            vizdoom.Button.TURN_RIGHT
        ],
        "frame_processor": lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
        "num_players": 3,
        "max_episode_length": 2000,
        "frame_skip": 4,
        "save_occupancy_maps": save_occupancy_maps,
    }

    config_train = base_config.copy()
    config_train["game_port"] = "5031"

    config_eval = base_config.copy()
    config_eval["game_port"] = "5032"

    threads = []
    for player_id in range(base_config["num_players"]):
        player_thread = Thread(
            target=run_single_agent,
            args=(
                player_id,
                config_train,
                config_eval,
                total_timesteps,
                eval_freq,
                n_eval_episodes,
            )
        )
        player_thread.start()
        threads.append(player_thread)

    for process in threads:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a distributed vision RL Agent in ViZDoom"
    )

    parser.add_argument(
        "--save_occupancy_maps",
        action="store_true",
        help="Saves the occupancy maps for each episode"
    )
    parser.add_argument(
        "--window_visible",
        action="store_true",
        help="Shows the video for each agent"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=40000,
        help="Total number of timesteps to train for"
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Total number of timesteps to train for"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5000,
        help="Total number of timesteps to train for"
    )

    args = parser.parse_args()
    run(
        args.save_occupancy_maps,
        args.window_visible,
        args.timesteps,
        args.eval_freq,
        args.n_eval_episodes,
    )
