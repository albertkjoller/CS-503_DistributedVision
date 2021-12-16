import argparse
from pathlib import Path
from typing import List, Tuple, Callable

import cv2 as cv2
import gym as gym
import numpy as np

from vizdoom import vizdoom
from gym.spaces import Space, Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common import callbacks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


class BaselineEnv(gym.Env):
    """ Environment used for the baseline """

    def __init__(
        self,
        game: vizdoom.DoomGame,
        frame_processor: Callable,
        frame_skip: int
    ):
        super().__init__()
        self.game = game
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip

        print("Environment parameters:")
        print(f"  * Frame skip: {self.frame_skip}")

        # Occupancy map
        self.map_x_boundaries = 160.0, 1120.0  # (min_x, max_x)
        self.map_y_boundaries = -704.0, 128.0  # (min_y, max_y)
        self.map_shape = (
            self.map_x_boundaries[1] - self.map_x_boundaries[0],
            self.map_y_boundaries[1] - self.map_y_boundaries[0]
        )
        self.occupancy_map_shape = (36, 24)  # (width, height) (36, 24)
        self.reset_occupancy_map()

        # The actions that can be taken
        self.action_space: Space = Discrete(game.get_available_buttons_size())
        self.possible_actions: list[list[int]] = np.eye(self.action_space.n).tolist()

        # The height, width and number of channels used by the game
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        # The height, width and number of channels passed to PPO
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        # The observation space. In our case RGB-D box with shape (4, W, H). The images returned by Vizdoom are uint8.
        self.observation_space: Space = Box(
            low=0,
            high=255,
            shape=(new_h, new_w, new_c + 1),
            dtype=np.uint8
        )

        # Empty frame for when the game is done
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # The current state of the game 
        self.state = self.empty_frame

        # The number of episodes that have been shown
        self.epsiode_number = 0

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        total_reward = 0
        for _ in range(self.frame_skip):
            # There is only one reward for the entire match, not one per agent
            step_reward = self.game.make_action(self.possible_actions[action], 1)
            total_reward += step_reward
            if self.game.is_episode_finished():
                break
        
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)
        return self.state, total_reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.reset_occupancy_map()
        self.epsiode_number += 1
        self.game.set_seed(self.epsiode_number)
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

        channels = []
        
        actor_state = self.game.get_state()
        if actor_state is None or actor_state.screen_buffer is None:
            print("Found none state: had to return empty frame")
            return self.empty_frame
        channels.append(
            self.frame_processor(actor_state.screen_buffer)
        )

        channels.append(
            np.expand_dims(self.frame_processor(self.occupancy_map), 2)
        )
        
        return np.concatenate(
            channels,
            axis=2
        )
    
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


def create_env(
    config_file_path: Path = Path("../setting/settings.cfg"),
    screen_resolution: vizdoom.ScreenResolution = vizdoom.ScreenResolution.RES_320X240,
    window_visible: bool = True,
    buttons: List[vizdoom.Button] = [vizdoom.Button.MOVE_FORWARD, vizdoom.Button.TURN_LEFT, vizdoom.Button.TURN_RIGHT],
    frame_processor: Callable = lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    max_episode_length: int = 2000,
    frame_skip: int = 4
) -> gym.Env:
    """"""
    game = vizdoom.DoomGame()

    game.load_config(str(config_file_path))

    # IMPORTANT
    game.set_seed(0)

    game.set_window_visible(window_visible)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.set_episode_timeout(max_episode_length)
    game.init()

    print("Set up game:")
    print(f"  * Max episode length: {max_episode_length}")
    print(f"  * Buttons:            {buttons}")
    print(f"  * Screen Resolution:  {screen_resolution}")

    return BaselineEnv(
        game,
        frame_processor,
        frame_skip
    )


def create_vec_env(eval: bool = False, **kwargs) -> VecTransposeImage:
    """"""
    if eval:
        return VecTransposeImage(DummyVecEnv([lambda: Monitor(create_env(**kwargs))]))

    return VecTransposeImage(DummyVecEnv([lambda: create_env(**kwargs)]))


def create_agent(env, **kwargs):
    """"""
    return PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=4096,
        batch_size=32,
        learning_rate=1e-4,
        tensorboard_log='logs/tensorboard',
        verbose=1,
        seed=0,
        **kwargs
    )


def run(
    window_visible: bool,
    total_timesteps: int,
    n_eval_episodes: int,
    eval_freq: int,
):

    # Configuration parameters
    config = {
        "config_file_path": Path("../setting/settings.cfg"),
        "screen_resolution": vizdoom.ScreenResolution.RES_320X240,
        "window_visible": window_visible,
        "buttons": [
            vizdoom.Button.MOVE_FORWARD,
            vizdoom.Button.TURN_LEFT,
            vizdoom.Button.TURN_RIGHT
        ],
        "frame_processor": lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
        "max_episode_length": 2000,
        "frame_skip": 4,
    }


    # Create training and evaluation environments.
    training_env, eval_env = create_vec_env(**config), create_vec_env(eval=True, **config)

    # Create the agent
    agent = create_agent(training_env)

    # Define an evaluation callback that will save the model when a new reward record is reached.
    evaluation_callback = callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        log_path='logs/evaluations/ppo_baseline',
        best_model_save_path='logs/models/ppo_baseline'
    )

    # Play!
    model = agent.learn(
        total_timesteps=total_timesteps,
        tb_log_name='ppo_baseline',
        callback=evaluation_callback
    )
    model.save('logs/models/ppo_baseline_final')

    # To view logs, run in another directory:
    #   tensorboard --logdir logs/tensorboard
    # And go to http://localhost:6006/ in Firefox or Chrome

    training_env.close()
    eval_env.close()


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
        default=250000,
        help="Total number of timesteps to train for"
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate on"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=4*4096,
        help="Number of timesteps between evaluation"
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        help="If training should continue from a trained model, pass the path to the file here."
    )

    args = parser.parse_args()
    run(
        args.window_visible,
        args.timesteps,
        args.n_eval_episodes,
        args.eval_freq,
    )

