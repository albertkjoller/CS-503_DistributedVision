# CS-503_DistributedVision
Object navigation by Reinforcement Learning agents using Distributed Vision. Done as a project in the course CS-503 - "Visual Intelligence: machines & minds" at EPFL (fall 2021).

![image](https://user-images.githubusercontent.com/57899625/144043367-204499ab-c7dc-4ffe-9e8d-37db83b7d00b.png)

## Project description
The overall research question is:
- Can a reinforcement learning agent usign distributed vision (multiple subagents) learn to navigate to an object in a maze faster and better than three regular agents or not?

The motivation is twofold; 1) to investigate whether the human cooperative behavirour of knowledge-sharing for solving certain tasks can succesfully be transfered to our understanding of intelligent agents and 2) to get a deeper understanding and know-how of Embodied Intelligence and the use of Computer Vision in Reinforcement Learning.

This task of Embodied Intelligence is adressed by creating two sets of reinforcement learning agents using visual features and evaluating them afterwards. The two sets can be thought of as 1) a centralized learner using the shared knowledge of multiple distributed camera-agents operating synchronously in the same maze and 2) multiple individual visual agents operating in the same maze without communicating. In both cases the agents have access to a simple memory-mechanism of their previous locations in the maze, which is implemented as a simple, binary occupancy map. 

The agents are operating in a [ViZDoom environment](https://github.com/mwydmuch/ViZDoom) - a simple way to create and train simulation based Reinforcement Learning agents. Our implementation uses the PPO implementation of [`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for which reason we wrap our VizDoom game environment such that it is compatible with the `stable-baselines3` implementation. 


## Repository overview
The repository should be self-explanatory. Within the `content` directory, a `setting` directory holding the training maps and the configuration-file is located. All the reinforcement learning, the models and the wrapping of the VizDoom environment is done in the `stable-baselines3` directory. Files related to the generation and usage of the occupancy map are located in the `content` directory along with a spectator instance of the environment (taken from [ViZDoom](https://github.com/mwydmuch/ViZDoom)), that can be used to explore the maze manually and without the usage of Reinforcement Learning and learning.
