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


class DistributedVisionEnv(gym.Env):
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
            shape=(new_h, new_w, new_c),
            dtype=np.uint8
        )

        # Empty frame for when the game is done
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # The current state of the game 
        self.state = self.empty_frame

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._get_frame()
        return self.state

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> np.ndarray:
        return (
            self.frame_processor(self.game.get_state().screen_buffer)
            if not done else self.empty_frame
        )


def create_env(
    config_file_path: Path = Path("../setting/settings.cfg"),
    screen_resolution: vizdoom.ScreenResolution = vizdoom.ScreenResolution.RES_320X240,
    window_visible: bool = True,
    buttons: List[vizdoom.Button] = [vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT, vizdoom.Button.ATTACK],
    frame_processor: Callable = lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    frame_skip: int = 4
) -> gym.Env:
    """"""
    game = vizdoom.DoomGame()
    game.load_config(str(config_file_path))
    game.set_window_visible(window_visible)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.init()

    return DistributedVisionEnv(
        game,
        frame_processor,
        frame_skip
    )


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


# Configuration parameters
config = {
    "config_file_path": Path("../setting/settings.cfg"),
    "screen_resolution": vizdoom.ScreenResolution.RES_320X240,
    "window_visible": True,
    "buttons": [
        vizdoom.Button.MOVE_FORWARD,
        vizdoom.Button.TURN_LEFT,
        vizdoom.Button.TURN_RIGHT
    ],
    "frame_processor": lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    "frame_skip": 4,
}


# Create training and evaluation environments.
training_env, eval_env = create_vec_env(**config), create_vec_env(eval=False, **config)

# Create the agent
agent = create_agent(training_env)

# Define an evaluation callback that will save the model when a new reward record is reached.
evaluation_callback = callbacks.EvalCallback(
    eval_env,
    n_eval_episodes=10,
    eval_freq=5000,
    log_path='logs/evaluations/ppo_baseline',
    best_model_save_path='logs/models/ppo_baseline'
)

# Play!
agent.learn(
    total_timesteps=40000,
    tb_log_name='ppo_baseline',
    callback=evaluation_callback
)

# To view logs, run in another directory:
#   tensorboard --logdir logs/tensorboard
# And go to http://localhost:6006/ in Firefox or Chrome

training_env.close()
eval_env.close()
