from pathlib import Path
from typing import List, Tuple

import gym as gym
import numpy as np

from vizdoom import vizdoom
from gym.spaces import Space, Discrete, Box
from stable_baselines3 import PPO


class DoomConfiguration:

    def __init__(
        self,
        config_file_path: Path,
        buttons: List[vizdoom.Button],
        window_visible: bool,
        screen_resolution: vizdoom.ScreenResolution,
    ) -> None:
        self.game = vizdoom.DoomGame()
        self.game.load_config(str(config_file_path))
        self.game.set_window_visible(window_visible)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.set_screen_format(vizdoom.ScreenFormat.RGB24)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(False)  # What is labels
        self.game.set_available_buttons(buttons)

        self.actions = self._get_actions()
        self.actions_size = len(self.actions)

        self.game.init()

    def _get_actions(self):
        n = self.game.get_available_buttons_size()
        actions = [n * [0] for _ in range(n)]
        for i in range(n):
            actions[i][i] = 1
        return actions


class DistributedVisionEnv(gym.Env):

    def __init__(self, env_config: dict):
        super().__init__()
        # The height and width of the game screen
        self.height = env_config["height"]
        self.width = env_config["width"]
        self.environment = DoomConfiguration(
            env_config["config_file_path"],
            env_config["buttons"],
            env_config["window_visible"],
            env_config["screen_resolution"],
        )

        # The actions that can be taken. In our case, that will be (forward, backward, left, right, move_left, move_right)
        self.action_space: Space = Discrete(self.environment.actions_size)

        # The observation space. In our case RGB-D box with shape (4, W, H). The images returned by Vizdoom are uint8.
        self.observation_space: Space = Box(
            low=0,
            high=255,
            shape=(env_config['height'], env_config['width'], 4),
            dtype=np.uint8
        )

    def reset(self) -> Space:
        self.environment.game.new_episode()
        state = self.environment.game.get_state()
        return self._vizdoom_state_to_rgbd(state)

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        reward: float = self.environment.game.make_action(self.environment.actions[action])
        done: bool = self.environment.game.is_episode_finished()

        if done:
            state: np.ndarray = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        else:
            raw_state: vizdoom.GameState = self.environment.game.get_state()
            state: np.ndarray = self._vizdoom_state_to_rgbd(raw_state)

        info: dict = {}
        return state, reward, done, info

    def _vizdoom_state_to_rgbd(self, state: vizdoom.GameState) -> np.ndarray:
        """"""
        rgb_height_width_channels: np.ndarray = state.screen_buffer
        d: np.ndarray = state.depth_buffer

        d = d.reshape(self.height, self.width, 1)
        rgbd = np.concatenate([rgb_height_width_channels, d], axis=2)

        return rgbd


env_config = {
    "height": 120,
    "width": 160,
    "config_file_path": Path("../setting/settings.cfg"),
    "buttons": [
        #vizdoom.Button.MOVE_LEFT,
        #vizdoom.Button.MOVE_RIGHT,
        #vizdoom.Button.ATTACK,
        vizdoom.Button.MOVE_FORWARD,
        vizdoom.Button.TURN_LEFT,
        vizdoom.Button.TURN_RIGHT
    ],
    "window_visible": True,
    "screen_resolution": vizdoom.ScreenResolution.RES_160X120,
}

env = DistributedVisionEnv(env_config)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_zdoom")

# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
