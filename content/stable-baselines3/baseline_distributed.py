from os import read
from pathlib import Path
from time import sleep
from typing import List, Tuple, Callable
from threading import Thread

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
        games: List[vizdoom.DoomGame],
        frame_processor: Callable,
        frame_skip: int
    ):
        super().__init__()
        print("Initializing DistributedVisionEnvironment")

        self.step_count = 0

        self.games = games
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip

        # The actions that can be taken
        self._game_button_sizes = [
            game.get_available_buttons_size()
            for game in games
        ]

        # All combinations of actions possible
        self.action_space: Space = Discrete(sum(self._game_button_sizes))
        self.possible_actions: list[list[int]] = []
        for agent_id in range(len(self.games)):

            num_possible_actions = self._game_button_sizes[agent_id]
            agent_actions = np.eye(num_possible_actions).tolist()

            if len(self.possible_actions) == 0:
                self.possible_actions = agent_actions
            else:
                new_action_combinatations = []
                for action in self.possible_actions:
                    for new_agent_action in agent_actions:
                        new_action_combinatations.append(
                            action + new_agent_action
                        )
                self.possible_actions = new_action_combinatations

        print(f"  * Action Space Size: {sum(self._game_button_sizes)}")
        print(f"  * Possible Actions: {sum(self._game_button_sizes)}")
        for action in self.possible_actions:
            print(f"    - {action}")

        out_h, out_w = None, None
        out_channels = 0
        for game in self.games:
            h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
            new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape

            if out_h is None:
                out_h = new_h

            if out_w is None:
                out_w = new_w

            assert out_h == new_h, "All agents must have the same output frame shape"
            assert out_w == new_w, "All agents must have the same output frame shape"

            out_channels += new_c

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

    def _make_agent_action(self, step_count, agent_id, game, action: list, rewards: list, is_done: list):
        agent_reward = game.make_action(action, self.frame_skip)
        rewards[agent_id] = agent_reward
        is_done[agent_id] = game.is_episode_finished()
        #print(f"  * Step {step_count}: AGENT {agent_id} taking action {action}")

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        self.step_count += 1

        action_combination = self.possible_actions[action]
        agent_start_index = 0
        agent_actions = []
        for i in range(3):
            agent_end_index = agent_start_index + self._game_button_sizes[i]
            agent_actions.append(action_combination[agent_start_index:agent_end_index])
            agent_start_index = agent_end_index
        #print(f"Step {self.step_count} with actions {agent_actions}")

        game_rewards = [0, 0, 0]
        game_done = [False, False, False]
        threads = []

        for i, game in enumerate(self.games):
            agent_action = agent_actions[i]
            thread = Thread(
                target=self._make_agent_action,
                args=(
                    self.step_count,
                    i,
                    game,
                    agent_action,
                    game_rewards,
                    game_done
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        reward = sum(game_rewards)
        done = sum(game_done) > 0
        self.state = self._get_frame(done)
        return self.state, reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        print("Resetting")
        threads = []
        for game in self.games:
            thread = Thread(target=game.new_episode)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        self.state = self._get_frame()
        return self.state

    def close(self) -> None:
        print("Closing")
        threads = []
        for game in self.games:
            thread = Thread(target=game.close)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> np.ndarray:
        if done:
            return self.empty_frame
        
        game_states = []
        for game in self.games:   
            actor_state = game.get_state()
            if actor_state is None or actor_state.screen_buffer is None:
                print("Found none state: had to return empty frame")
                return self.empty_frame

            game_states.append(
                self.frame_processor(actor_state.screen_buffer)
            )
        
        return np.concatenate(
            game_states,
            axis=2
        )


def add_player(
    game: vizdoom.DoomGame(),
    port: str,
    num_players: int,
    player_id: int,
    host: bool,
    episode_length: int,
    config_file_path: Path = Path("../setting/settings.cfg"),
    screen_resolution: vizdoom.ScreenResolution = vizdoom.ScreenResolution.RES_320X240,
    window_visible: bool = True,
    buttons: List[vizdoom.Button] = [vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT, vizdoom.Button.ATTACK],
) -> vizdoom.DoomGame:
    game.load_config(str(config_file_path))
    game.set_window_visible(window_visible)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.set_episode_timeout(episode_length)

    if host:
        game.add_game_args(
            f"-host {num_players}" +
            f" -port {port}" +
            " -netmode 0" + 
            f" +timelimit {episode_length}" +
            " +sv_spawnfarthest 1" +
            " +name Player0" +
            " +colorset 0"
        )
    else:
        game.add_game_args(
            "-join 127.0.0.1" +
            f" -port {port}" +
            f" +name Player{player_id}" +
            f" +colorset {player_id}"
        )

    game.init()
    print(f"INITIALIZED PLAYER {player_id}")

    return game


def create_env(
    config_file_path: Path = Path("../setting/settings.cfg"),
    port: str = "5029",
    screen_resolution: vizdoom.ScreenResolution = vizdoom.ScreenResolution.RES_320X240,
    window_visible: bool = True,
    buttons: List[vizdoom.Button] = [vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT, vizdoom.Button.ATTACK],
    frame_processor: Callable = lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    num_players: int = 3,
    max_episode_length: int = 500,
    frame_skip: int = 1
) -> gym.Env:
    """"""
    aux_games = []
    threads = []
    for player_id in range(1, num_players):
        game = vizdoom.DoomGame()
        player_thread = Thread(
            target=add_player,
            args=(
                game,
                port,
                num_players,
                player_id,
                False,
                max_episode_length,
            ),
            kwargs={
                "config_file_path": config_file_path,
                "screen_resolution": screen_resolution,
                "window_visible": window_visible,
                "buttons": buttons,
            }
        )
        player_thread.start()
        aux_games.append(game)
        threads.append(player_thread)

    host_game = vizdoom.DoomGame()
    add_player(
        host_game,
        port,
        num_players,
        0,
        True,
        max_episode_length,
        config_file_path=config_file_path,
        screen_resolution=screen_resolution,
        window_visible=window_visible,
        buttons=buttons
    )

    for thread in threads:
        thread.join()

    return DistributedVisionEnv(
        [host_game, aux_games[0], aux_games[1]],
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
base_config = {
    "port": "5029",
    "config_file_path": Path("../setting/settings.cfg"),
    "screen_resolution": vizdoom.ScreenResolution.RES_320X240,
    "window_visible": True,
    "buttons": [
        vizdoom.Button.MOVE_FORWARD,
        vizdoom.Button.TURN_LEFT,
        vizdoom.Button.TURN_RIGHT
    ],
    "frame_processor": lambda frame: cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
    "num_players": 3,
    "max_episode_length": 500,
    "frame_skip": 1,
}

config_train = base_config.copy()
config_train["port"] = "5029"

config_eval = base_config.copy()
config_eval["port"] = "5030"


# Create training and evaluation environments.
training_env = create_vec_env(**config_train)
eval_env = create_vec_env(eval=False, **config_eval)

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