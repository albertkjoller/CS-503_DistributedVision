# DistViz - a Computer Vision based approach for Reinforcement Learning with Distributed Vision
Object navigation by Reinforcement Learning agents using Distributed Vision. Done as a project in the course [CS-503 - "Visual Intelligence: machines & minds"](https://edu.epfl.ch/coursebook/fr/visual-intelligence-machines-and-minds-CS-503) at EPFL (fall 2021). The project report is included in the repository.

<img src="./distributed_siamese.gif" alt="" width="1000">

## Project description
The overall research question is:
- Can a reinforcement learning agent usign distributed vision (multiple subagents) learn to navigate to an object in a maze faster and better than three regular agents or not?

The motivation is twofold; 1) to investigate whether the human cooperative behavirour of knowledge-sharing for solving certain tasks can succesfully be transfered to our understanding of intelligent agents and 2) to get a deeper understanding and know-how of Embodied Intelligence and the use of Computer Vision in Reinforcement Learning.

This task of Embodied Intelligence is adressed by creating two sets of reinforcement learning agents using visual features and evaluating them afterwards. The two sets can be thought of as 1) a centralized learner using the shared knowledge of multiple distributed camera-agents operating synchronously in the same maze and 2) multiple individual visual agents operating in the same maze without communicating. In both cases the agents have access to a simple memory-mechanism of their previous locations in the maze, which is implemented as a simple, binary occupancy map. 

The agents are operating in a [ViZDoom environment](https://github.com/mwydmuch/ViZDoom) - a simple way to create and train simulation based Reinforcement Learning agents. Our implementation uses the PPO implementation of [`stable-baselines3`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for which reason we wrap our VizDoom game environment such that it is compatible with the `stable-baselines3` implementation. 


## Repository overview
The repository should be self-explanatory. Within the `content` directory, a `setting` directory holding the training maps and the configuration-file is located. All the reinforcement learning, the models and the wrapping of the VizDoom environment is done in the `stable-baselines3` directory. Files related to the generation and usage of the occupancy map are located in the `content` directory along with a spectator instance of the environment (taken from [ViZDoom](https://github.com/mwydmuch/ViZDoom)), that can be used to explore the maze manually and without the usage of Reinforcement Learning and learning.

### Content

#### Installation

To create an Azure environment with the correct packages installed, the following Dockerfile can be used.
```Dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20211124.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/pytorch-1.10

# Install dependencies for vizdoom
RUN apt-get update -y
RUN apt install -y \
    cmake \
    libboost-all-dev \
    libsdl2-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libjpeg-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    zlib1g-dev \
    timidity \
    tar \
    nasm


# Create conda environment with AI Gym and Pytorch
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=20.2.4 \
    pytorch=1.10.0 \
    torchvision=0.11.1 \
    torchaudio=0.10.0 \
    cudatoolkit=11.1.1 \
    nvidia-apex=0.1.0 \
    gxx_linux-64 \
    gym \
    -c anaconda -c pytorch -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies, including vizdoom and opencv
RUN pip install 'matplotlib>=3.3,<3.4' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.63' \
                'pandas>=1.3,<1.4' \
                'scipy>=1.5,<1.8' \
                'numpy>=1.10,<1.22' \
                'ipykernel~=6.0' \
                'azureml-core==1.36.0.post2' \
                'azureml-defaults==1.36.0' \
                'azureml-mlflow==1.36.0' \
                'azureml-telemetry==1.36.0' \
                'tensorboard==2.6.0' \
                'tensorflow-gpu==2.6.0' \
                'onnxruntime-gpu>=1.7,<1.10' \
                'horovod==0.23' \
                'future==0.18.2' \
                'torch-tb-profiler==0.3.1' \
                'vizdoom' \
                'opencv-python'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
```

#### Training

To train the models, from the `content/` directory, run:
```Bash
# Baseline
python stable-baselines3/baseline.py --timesteps 3000000 --n_eval_episodes 10 --eval_freq 8192
# Distributed Vision
python stable-baselines3/baseline_distributed3.py --timesteps 3000000 --n_eval_episodes 10 --eval_freq 8192
# Distributed Vision with Siamese Architecture
python stable-baselines3/baseline_distributed4.py --timesteps 3000000 --n_eval_episodes 10 --eval_freq 8192
```

The logs and trained models are stored in `content/logs/`.

#### Evaluation

To evaluate the models, from the `content/` directory, run:
```Bash
# Baseline
python stable-baselines3/evaluate_baseline.py path/to/model --window_visible
# Distributed Vision (all models)
python stable-baselines3/evaluate_baseline_distributed.py path/to/model --window_visible
```

#### A Note on Reproducibility

We have reproducibility issues in our code. Despite setting randomness seeds for all packages we use, it seems we cannot get the same results when running the code twice. The issue seems to be coming by the seed set for ViZDoom.

Not only that, but due to the poorly coded network architecture in ViZDoom, when running the distributed vision models the code can enter a state where all agents are waiting for each other in a locked state and we need to restart the program. Of course, these points are random and not reproducible.
