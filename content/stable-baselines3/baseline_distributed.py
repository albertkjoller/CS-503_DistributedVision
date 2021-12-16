import argparse

from os import read
from pathlib import Path
from time import sleep
from typing import List, Tuple, Callable, Optional
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


# TODO: 1 vs 3 occupancy maps


class DistributedVisionEnv(gym.Env):
    """ Environment used for the baseline """

    def __init__(
        self,
        games: List[vizdoom.DoomGame],
        frame_processor: Callable,
        frame_skip: int,
        save_occupancy_maps: bool,
        base_seed: int = 0
    ):
        super().__init__()
        print("Initializing DistributedVisionEnvironment")
        self.num_actors = len(games)
        self.games = games
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip
        self.save_occupancy_maps = save_occupancy_maps

        self.random_base_seed = base_seed
        self.epsiode_number = 0

        self.map_x_boundaries = 160.0, 1120.0  # (min_x, max_x)
        self.map_y_boundaries = -704.0, 128.0  # (min_y, max_y)
        self.map_shape = (
            self.map_x_boundaries[1] - self.map_x_boundaries[0],
            self.map_y_boundaries[1] - self.map_y_boundaries[0]
        )
        self.occupancy_map_shape = (36, 24)  # (width, height) (36, 24)
        self.reset_occupancy_map()

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
        print(f"  * Possible Actions: {len(self.possible_actions)}")
        for action in self.possible_actions:
            print(f"    - {action}")

        out_h, out_w = None, None
        out_channels = 1  # start at 1 for occupancy map
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

    def reset_occupancy_map(self):
        """"""
        self.occupancy_map = np.zeros(self.occupancy_map_shape, dtype=np.uint8)

    def update_occupancy_map(self, game: vizdoom.DoomGame) -> np.ndarray:
        pos_x = game.get_game_variable(vizdoom.GameVariable.POSITION_X)
        pos_y = game.get_game_variable(vizdoom.GameVariable.POSITION_Y)

        grid_x = min(
            int(self.occupancy_map_shape[0] * ((pos_x - self.map_x_boundaries[0])/self.map_shape[0])),
            self.occupancy_map_shape[0] - 1
        )
        grid_y = min(
            int(self.occupancy_map_shape[1] * ((pos_y - self.map_y_boundaries[0])/self.map_shape[1])),
            self.occupancy_map_shape[1] - 1
        )

        self.occupancy_map[grid_x, grid_y] = 255  # Max of np.unint8

    def _make_agent_action(self, agent_id, game, action: list, rewards: list, is_done: list):
        agent_reward = game.make_action(action, 1)  # Forced to 1 as with frame_skip above 1 network fails
        self.update_occupancy_map(game)

        # If we're trying to move forward and bump into a wall, negative reward
        # Should we just have a positive reward for velocity?
        #vx = game.get_game_variable(vizdoom.GameVariable.VELOCITY_X)
        #vy = game.get_game_variable(vizdoom.GameVariable.VELOCITY_Y)
        #velocity = np.sqrt(vx ** 2 + vy ** 2)

        # TODO: Should make sure action[0] is move forward
        #if action[0] == 1 and velocity < 1:
        #    agent_reward -= 0.0005
        rewards[agent_id] = agent_reward
        is_done[agent_id] = game.is_episode_finished()

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        """
        Responsible for calling reset when the game is finished
        """
        action_combination = self.possible_actions[action]
        agent_start_index = 0
        agent_actions = []
        for i in range(self.num_actors):
            agent_end_index = agent_start_index + self._game_button_sizes[i]
            agent_actions.append(action_combination[agent_start_index:agent_end_index])
            agent_start_index = agent_end_index

        frame_skip_rewards = []
        frame_skip_done = []
        for i in range(self.frame_skip):
            # There is only one reward for the entire match, not one per agent
            threads = []
            for i, game in enumerate(self.games):
                agent_action = agent_actions[i]
                thread = Thread(
                    target=game.make_action,
                    args=(agent_action, 1),
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join(timeout=2.0)
            
            frame_skip_rewards.append(self.games[0].get_last_reward())
            frame_skip_done.append(self.games[0].is_episode_finished())

            if frame_skip_done[-1]:
                break
        
        reward = sum(frame_skip_rewards)
        done = sum(frame_skip_done) > 0
        self.state = self._get_frame(done)
        return self.state, reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        print("Resetting")
        if self.save_occupancy_maps:
            plt.figure()
            plt.title("Occupancy map")
            plt.imshow(self.occupancy_map)
            plt.savefig(f"../results/occ_map_episode_{self.epsiode_number}.jpg")
            plt.close()

        self.epsiode_number += 1
        self.reset_occupancy_map()

        threads = []
        for game in self.games:
            thread = Thread(target=game.new_episode)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        for player_id, game in enumerate(self.games):
            game.set_seed(self.random_base_seed + 1000 * player_id + self.epsiode_number)

        self.state = self._get_frame()
        print("Done Resetting")
        return self.state

    def close(self) -> None:
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

        game_states.append(
            np.expand_dims(self.frame_processor(self.occupancy_map), 2)
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

    # IMPORTANT
    game.set_seed(int(port) + 1000 * player_id)

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
    frame_skip: int = 1,
    save_occupancy_maps: bool = False,
    base_seed: int = 0,
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
        frame_skip,
        save_occupancy_maps,
        base_seed=base_seed,
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
        n_epochs=10,
        n_steps=4096,
        batch_size=32,
        learning_rate=1e-4,
        tensorboard_log='logs/tensorboard',
        verbose=1,
        seed=0,
        **kwargs
    )


def run(
    save_occupancy_maps: bool,
    window_visible: bool,
    total_timesteps: int,
    n_eval_episodes: int,
    eval_freq: int,
    pretrained_model: Optional[str],
):

    np.random.seed(0)

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
    config_train["port"] = "5030"
    config_train["base_seed"] = 0

    config_eval = base_config.copy()
    config_eval["port"] = "5031"
    config_eval["base_seed"] = 100000

    # Create training and evaluation environments.
    training_env = create_vec_env(**config_train)
    eval_env = create_vec_env(eval=True, **config_eval)

    # Create the agent
    agent = create_agent(training_env)
    if pretrained_model is not None:
        print(f"Loading pretrained model from {pretrained_model}")
        agent.load(pretrained_model)

    # Define an evaluation callback that will save the model when a new reward record is reached.
    evaluation_callback = callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        log_path='logs/evaluations/ppo_distributed_vision',
        best_model_save_path='logs/models/ppo_distributed_vision'
    )

    # Play!
    """
    :param total_timesteps: The total number of samples (env steps) to train on
    :param callback: callback(s) called at every step with state of the algorithm.
    :param log_interval: The number of timesteps before logging.
    :param tb_log_name: the name of the run for TensorBoard logging
    :param eval_env: Environment that will be used to evaluate the agent
    :param eval_freq: Evaluate the agent every eval_freq timesteps (this may vary a little)
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param eval_log_path: Path to a folder where the evaluations will be saved
    :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
    :return: the trained model
    """
    model = agent.learn(
        total_timesteps=total_timesteps,
        tb_log_name='ppo_distributed_vision',
        #eval_env=eval_env,  # Don't know if I need to do this
        #eval_freq=total_timesteps,  # Don't know if I need to do this
        callback=evaluation_callback,
    )
    model.save('ppo_distributed_vision')


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
        args.save_occupancy_maps,
        args.window_visible,
        args.timesteps,
        args.n_eval_episodes,
        args.eval_freq,
        args.pretrained_model,
    )
