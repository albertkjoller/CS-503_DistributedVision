import argparse

from enum import Enum
from os import read
from pathlib import Path
from time import sleep
from typing import Any, List, Tuple, Callable, Optional
from multiprocessing import cpu_count, Process, Pipe

import cv2 as cv2
import gym as gym
from gym.core import Env
import matplotlib.pyplot as plt
import numpy as np

from vizdoom import vizdoom as vzd
from gym.spaces import Space, Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common import callbacks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from baseline_distributed_custom_feature_extractor import SiameseCNN, SiameseResNetCNN


# python stable-baselines3/evaluate_baseline_distributed.py results/logs/final_models/ppo_distributed_vision3_azure_3m/ppo_distributed_vision3_final --window_visible
# python stable-baselines3/evaluate_baseline_distributed.py results/logs/final_models/ppo_distributed_vision2_1/ppo_distributed_vision2_final --window_visible

def frame_processor(frame):
    return cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)


class EnvironmentAction(Enum):
    TAKE_STEP = 0
    RESET = 1
    CLOSE = 2


class AgentPosition:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class DistributedAgent:

    def __init__(
        self,
        player_id: int,
        num_players: int,
        game_port: int,
        game_creator: Callable,
        game_creator_kwargs: dict,
        max_episode_length: int,
        communication_pipe,
    ):
        self.player_id = player_id
        self.num_players = num_players
        self.game_port = game_port
        self.game_creator = game_creator
        self.game_creator_kwargs = game_creator_kwargs
        self.max_episode_length = max_episode_length
        self.communication_pipe = communication_pipe

        self.game: Optional[vzd.DoomGame] = None
        self.initialized = False

    def is_host(self):
        return self.player_id == 0

    def initialize_player(self):
        self.game = self.game_creator(self.player_id, **self.game_creator_kwargs)

        if self.is_host():
            self.game.add_game_args(
                f"-host {self.num_players}" +
                f" -port {self.game_port}" +
                " -netmode 0" + 
                f" +timelimit {self.max_episode_length}" +
                " +sv_spawnfarthest 1" +
                " +name Player0" +
                " +colorset 0"
            )
        else:
            self.game.add_game_args(
                "-join 127.0.0.1" +
                f" -port {self.game_port}" +
                f" +name Player{self.player_id}" +
                f" +colorset {self.player_id}"
            )
        
        self.game.init()
        self.initialized = True

    def run(self):
        self.initialize_player()
        while self.initialized:
            msg: Tuple[EnvironmentAction, Any] = self.communication_pipe.recv()
            todo = msg[0]
            
            if todo == EnvironmentAction.TAKE_STEP:
                agent_action = msg[1]
                step_reward, new_state = self.step(agent_action)
                agent_position = self.get_position()
                done = self.game.is_episode_finished()
                self.communication_pipe.send((step_reward, new_state, agent_position, done))
            elif todo == EnvironmentAction.RESET:
                seed = msg[1]
                self.set_seed(seed)
                self.reset()
                self.communication_pipe.send((True,))

            elif todo == EnvironmentAction.CLOSE:
                self.close()
                self.communication_pipe.send((True,))
                self.initialized = False
            else:
                raise RuntimeError(f"Agent {self.player_id} Given unknown action")

    def step(self, agent_action: List[int]) -> Tuple[float, Optional[np.ndarray]]:
        agent_reward = self.game.make_action(agent_action)
        agent_state = self.game.get_state()
        if agent_state is None or agent_state.screen_buffer is None:
            agent_view = None
        else:
            agent_view = agent_state.screen_buffer
        return agent_reward, agent_view

    def reset(self) -> None:
        self.game.new_episode()

    def close(self) -> None:
        self.game.close()
        self.initialized = False

    def set_seed(self, seed: int) -> None:
        self.game.set_seed(seed)
    
    def get_position(self) -> AgentPosition:
        if not self.initialized:
            raise RuntimeError("Called get_position on uninitialized Agent")

        return AgentPosition(
            self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        )


class DistributedVisionEnvironment(gym.Env):

    def __init__(
        self,
        num_players: int,
        game_port: int,
        game_creator: Callable[[int], vzd.DoomGame],
        game_creator_kwargs: dict,
        frame_processor: Callable,
        frame_skip: int,
        max_episode_length: int,
    ):
        self.num_players = num_players
        self.game_port = game_port
        self.game_creator = game_creator
        self.game_creator_kwargs = game_creator_kwargs
        self.frame_processor = frame_processor
        self.frame_skip = frame_skip
        self.max_episode_length = max_episode_length

        self.episode_number = 0

        # 3 buttons for each game
        self._game_button_sizes: List[int] = [3, 3, 3]
        self.action_space: Space = Discrete(sum(self._game_button_sizes))
        self.possible_actions: List[List[int]] = self.compute_possible_actions()
        print(f"  * Action Space Size: {sum(self._game_button_sizes)}")
        print(f"  * Possible Actions: {len(self.possible_actions)}")

        # Brute force this
        self.observation_space: Space = Box(
            low=0,
            high=255,
            shape=(120, 160, 10),
            dtype=np.uint8
        )

        # Empty frame for when the game is done
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        # The current state of the game 
        self.state = self.empty_frame

        self.initialize_occupancy_map()
        self.reset_occupancy_map()
        self.initialize_agents()

    def step(self, action: int) -> Tuple[Space, float, bool, dict]:
        action_combination = self.possible_actions[action]
        agent_start_index = 0
        agent_actions = []
        for i in range(self.num_players):
            agent_end_index = agent_start_index + self._game_button_sizes[i]
            agent_actions.append(action_combination[agent_start_index:agent_end_index])
            agent_start_index = agent_end_index

        rewards = []
        done = False
        for i in range(self.frame_skip):
            # Send actions to take
            for agent_pipe, action_action in zip(self.pipes, agent_actions):
                agent_pipe.send(
                    (
                        EnvironmentAction.TAKE_STEP,
                        action_action,
                    )
                )
            
            # Wait for results
            frame_reward = 0
            agent_states = []
            for agent_id, agent_pipe in enumerate(self.pipes):
                if not agent_pipe.poll(timeout=4.0): 
                    print(f"Network out of sync in step(), blocked on {agent_id}. Restarting all.")
                    self.restart_all_actors()
                    return self.state, 0, True, {}

                agent_reward, agent_state, agent_position, agent_done = agent_pipe.recv()
                frame_reward = agent_reward  # All agent rewards are the same
                agent_states.append(agent_state)
                self.update_occupancy_map(agent_position)
                done = done or agent_done
            
            rewards.append(frame_reward)
            if done:
                break
        
        reward = sum(rewards)
        self._update_state(done, agent_states)
        return self.state, reward, done, {}

    def reset(self):
        self.episode_number += 1

        for agent_id, agent_pipe in enumerate(self.pipes):
            agent_seed = 10000 * self.game_port + 1000 * agent_id + self.episode_number
            agent_pipe.send((
                    EnvironmentAction.RESET,
                    agent_seed,
                ))

        for agent_id, agent_pipe in enumerate(self.pipes):
            if not agent_pipe.poll(timeout=4.0):
                print(f"Network out of sync in reset(), blocked on {agent_id}. Restarting all.")
                self.restart_all_actors()
                return self.state
            
            _ = agent_pipe.recv()
        
        self._update_state(True, None)
        return self.state

    def close(self):
        for agent_pipe in self.pipes:
            agent_pipe.send((
                    EnvironmentAction.CLOSE,
                    None,
                ))

        for agent_pipe in self.pipes:
            _ = agent_pipe.recv()

    def restart_all_actors(self):
        for agent_process in self.agent_processes:
            agent_process.terminate()

        sleep(5)
        self.initialize_agents()
        self.reset()

    def _update_state(self, done: bool, agent_states: List[np.ndarray]) -> None:
        if done:
            self.state = self.empty_frame
            return
        
        states = []
        for agent_state in agent_states:   
            if agent_state is None:
                print("Found none state when not done: had to set empty frame")
                self.state = self.empty_frame
                return
            
            states.append(self.frame_processor(agent_state))
        states.append(np.expand_dims(self.frame_processor(self.occupancy_map), 2))
        self.state = np.concatenate(states, axis=2)

    def initialize_agents(self):
        communication_pipes = [Pipe() for _ in range(self.num_players)]
        parent_pipes = [pipe[0] for pipe in communication_pipes]
        child_pipes = [pipe[1] for pipe in communication_pipes]
        self._agent_pipes = child_pipes
        self.pipes = parent_pipes

        agents = [
            DistributedAgent(
                player_id,
                self.num_players,
                self.game_port,
                self.game_creator,
                self.game_creator_kwargs,
                self.max_episode_length,
                child_pipes[player_id],
            )
            for player_id in range(self.num_players)
        ]

        self.agent_processes: list[Process] = [
            Process(target=agent.run)
            for agent in agents
        ]

        for process in self.agent_processes:
            process.start()

        # Wait for agents to connect
        sleep(5)

    def initialize_occupancy_map(self):
        self.map_x_boundaries = 160.0, 1120.0  # (min_x, max_x)
        self.map_y_boundaries = -704.0, 128.0  # (min_y, max_y)
        self.map_shape = (
            self.map_x_boundaries[1] - self.map_x_boundaries[0],
            self.map_y_boundaries[1] - self.map_y_boundaries[0]
        )
        self.occupancy_map_shape = (36, 24)  # (width, height) (36, 24)

    def reset_occupancy_map(self):
        self.occupancy_map = np.zeros(self.occupancy_map_shape, dtype=np.uint8)

    def update_occupancy_map(self, agent_position: AgentPosition) -> np.ndarray:
        pos_x = agent_position.x
        pos_y = agent_position.y

        grid_x = min(
            int(self.occupancy_map_shape[0] * ((pos_x - self.map_x_boundaries[0])/self.map_shape[0])),
            self.occupancy_map_shape[0] - 1
        )
        grid_y = min(
            int(self.occupancy_map_shape[1] * ((pos_y - self.map_y_boundaries[0])/self.map_shape[1])),
            self.occupancy_map_shape[1] - 1
        )

        self.occupancy_map[grid_x, grid_y] = 255  # Max of np.unint8

    def compute_possible_actions(self) -> List[List[int]]:
        possible_actions = []
        for agent_id in range(self.num_players):
            num_possible_actions = self._game_button_sizes[agent_id]
            agent_actions = np.eye(num_possible_actions).tolist()

            if len(possible_actions) == 0:
                possible_actions = agent_actions
            else:
                new_action_combinatations = []
                for action in possible_actions:
                    for new_agent_action in agent_actions:
                        new_action_combinatations.append(
                            action + new_agent_action
                        )
                possible_actions = new_action_combinatations
        return possible_actions


def game_creator(
    player_id: int,
    config_file_path=None,
    window_visible=None,
    port=None,
    screen_resolution=None,
    buttons=None,
    max_episode_length=None,
) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(config_file_path))
    game.set_seed(int(port) + 1000 * player_id)
    game.set_window_visible(window_visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(screen_resolution)
    game.set_labels_buffer_enabled(False)  # What is labels
    game.set_available_buttons(buttons)
    game.set_episode_timeout(max_episode_length)
    return game


def create_env(
    config_file_path: Path = Path("../setting/settings.cfg"),
    port: str = "5029",
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_320X240,
    window_visible: bool = True,
    buttons: List[vzd.Button] = [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK],
    frame_processor: Callable = frame_processor,
    num_players: int = 3,
    max_episode_length: int = 500,
    frame_skip: int = 1,
) -> gym.Env:
    """"""

    game_creator_kwargs = {
        "config_file_path": config_file_path,
        "port": port,
        "window_visible": window_visible,
        "screen_resolution": screen_resolution,
        "buttons": buttons,
        "max_episode_length": max_episode_length,
    }

    return DistributedVisionEnvironment(
        num_players,
        int(port),
        game_creator,
        game_creator_kwargs,
        frame_processor,
        frame_skip,
        max_episode_length,
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
        policy_kwargs={'features_extractor_class': SiameseCNN},
        verbose=1,
        seed=0,
        **kwargs
    )


def evaluate(
    model_path: str,
    num_epsisodes: int,
    window_visible: bool,
    save_occupancy_maps: bool,
):
    np.random.seed(0)

    # Configuration parameters
    base_config = {
        "config_file_path": Path("setting/settings.cfg"),
        "screen_resolution": vzd.ScreenResolution.RES_320X240,
        "window_visible": window_visible,
        "buttons": [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT
        ],
        "frame_processor": frame_processor,
        "num_players": 3,
        "max_episode_length": 2000,
        "frame_skip": 4,
    }

    # Set the port
    base_config["port"] = "5090"

    # Create the environment and model
    env = create_env(**base_config)
    model = PPO.load(model_path)

    # Evaluate
    rewards = []
    steps = []
    for episode_num in range(num_epsisodes):
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

        rewards.append(episode_reward)
        steps.append(episode_steps)

        print()
        print(f"Done with episode {episode_num}!")
        print(f"  Reward: {episode_reward}")
        print(f"  Steps:  {episode_steps}")
        print()

    print(f"\n\n\n{50 * '-'}\n")
    print("Done Evaluating")

    print(f"\nRewards: {rewards}")
    print(f"Steps:   {steps}")

    print("\nMean:")
    print(f"  Rewards: {np.mean(rewards)}")
    print(f"  Steps:   {np.mean(steps)}")

    print("\nStandard Deviation:")
    print(f"  Rewards: {np.std(rewards)}")
    print(f"  Steps:   {np.std(steps)}")

    env.close()


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
        "--save_occupancy_maps",
        action="store_true",
        help="Saves the occupancy maps for each episode"
    )

    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.n_episodes,
        args.window_visible,
        args.save_occupancy_maps,
    )
