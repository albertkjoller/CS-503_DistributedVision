import argparse
from pathlib import Path
from typing import List, Tuple, Callable, Optional
from threading import Thread

import cv2 as cv2
import gym as gym
import numpy as np

from vizdoom import vizdoom
from gym.spaces import Space, Discrete, Box
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


GAME_PORT = 5055


# RUNNING THE CODE
#   python stable-baselines3/evaluate_baseline.py results/logs/final_models/ppo_baseline_azure_3m/baseline_final --window_visible


class BaselineEnv(gym.Env):
    """ Environment used for the baseline """

    def __init__(
        self,
        experiment_id: int,
        game: vizdoom.DoomGame,
        frame_processor: Callable,
        frame_skip: int,
        player_id: int,
    ):
        super().__init__()
        self.experiment_id = experiment_id
        self.game = game
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip
        self.player_id = player_id

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
        seed = 10000 * self.experiment_id + 1000 * self.player_id + self.epsiode_number
        self.game.set_seed(seed)
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
    experiment_id: int = 0,
    player_id: int = 0,
    config_file_path: Path = Path("setting/settings.cfg"),
    scenario_path: Optional[Path] = None,
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

    if scenario_path is not None:
        game.set_doom_scenario_path(str(scenario_path))

    game.set_seed(player_id)
    game.set_window_visible(window_visible)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.set_episode_timeout(max_episode_length)

    if player_id == 0:
        game.add_game_args(
            f"-host 3" +
            f" -port {GAME_PORT}" +
            " -netmode 0" + 
            f" +timelimit {max_episode_length}" +
            " +sv_spawnfarthest 1" +
            " +name Player0" +
            " +colorset 0"
        )
    else:
        game.add_game_args(
            "-join 127.0.0.1" +
            f" -port {GAME_PORT}" +
            f" +name Player{player_id}" +
            f" +colorset {player_id}"
        )
    game.init()

    print(f"Set up game {player_id}:")
    print(f"  * Max episode length: {max_episode_length}")
    print(f"  * Buttons:            {buttons}")
    print(f"  * Screen Resolution:  {screen_resolution}")

    return BaselineEnv(
        experiment_id,
        game,
        frame_processor,
        frame_skip,
        player_id,
    )


def create_vec_env(**kwargs) -> VecTransposeImage:
    """"""
    return VecTransposeImage(DummyVecEnv([lambda: create_env(**kwargs)]))



def single_agent_evaluation(num_episodes: int, config: dict, model: PPO, results: list):
    env = create_vec_env(**config)

    loop = range(num_episodes)
    if config["player_id"] == 0:
        loop = tqdm(loop)

    for episode_num in loop:
        obs = env.reset()
        
        episode_steps = 0
        episode_reward = 0
        episode_done = False
        while not episode_done:
            episode_steps += 1

            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_done = done
            
            env.render()

        results.append((episode_reward[0], episode_steps))


def evaluate(
    model_path: str,
    num_episodes: int,
    window_visible: bool,
    test_maze: bool,
):
    num_players = 3

    # Configuration parameters
    config = {
        "config_file_path": Path("setting/settings.cfg"),
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

    if test_maze:
        config["scenario_path"] = Path("setting/test.wad")

    model = PPO.load(model_path)

    rewards_and_steps = []
    player_threads = []
    for player_id in range(num_players):
        player_config = config.copy()
        player_config["experiment_id"] = 0
        player_config["player_id"] = player_id

        player_rewards = []

        player_thread = Thread(
            target=single_agent_evaluation,
            args=(num_episodes, player_config, model, player_rewards)
        )
        player_thread.start()

        player_threads.append(player_thread)
        rewards_and_steps.append(player_rewards)        
    
    for player_thread in player_threads:
        player_thread.join()

    rewards = [
        [
            round(episode_rewards_and_steps[0], 3)
            for episode_rewards_and_steps in player_rewards_and_steps
        ]
        for player_rewards_and_steps in rewards_and_steps
    ]
    steps = [
        [
            episode_rewards_and_steps[1]
            for episode_rewards_and_steps in player_rewards_and_steps
        ]
        for player_rewards_and_steps in rewards_and_steps
    ]

    rewards = rewards[0]
    steps_ = []
    for episode_num in range(num_episodes):
        steps_.append(min(
            steps[0][episode_num],
            steps[1][episode_num],
            steps[2][episode_num]
        ))
    steps = steps_
    
    print(f"\n\n\n{50 * '-'}\n{50 * '-'}\n{50 * '-'}\n")
    print("Done Evaluating")

    print(f"\nRewards: {rewards}")
    print(f"Steps:   {steps}")

    print("\nMean:")
    print(f"  Rewards: {np.mean(rewards)}")
    print(f"  Steps:   {np.mean(steps)}")

    print("\nStandard Deviation:")
    print(f"  Rewards: {np.std(rewards)}")
    print(f"  Steps:   {np.std(steps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a distributed vision RL Agent in ViZDoom"
    )

    parser.add_argument(
        "model_path",
        default="",

    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate on"
    )
    parser.add_argument(
        "--window_visible",
        action="store_true",
        help="Shows the video for each agent"
    )
    parser.add_argument(
        "--test_maze",
        action="store_true",
        help="Run the evaluation on the test maze"
    )

    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.n_episodes,
        args.window_visible,
        args.test_maze,
    )
