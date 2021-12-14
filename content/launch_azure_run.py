from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment


workspace = Workspace.from_config()
exp = Experiment(workspace, 'DVizTestRun')

env_name = "vizdoom"
env_image = "41170d9374bb44909ad2f76b01d40b32.azurecr.io/azureml/azureml_7acaf78c61ae240b65b61761a653b254"
env = Environment.from_docker_image(
    name=env_name,
    image=env_image,
)

relative_path_from_code = "Users/niels.poulsen/CS-503_DistributedVision/content/"
relative_path = "CS-503_DistributedVision/content/"

config = ScriptRunConfig(
    source_directory=relative_path,
    compute_target="RLTraining",
    script="stable-baselines3/baseline_distributed.py",
    environment=env,
    arguments=[
        '--timesteps', 200000,
        '--n_eval_episodes', 10,
        '--eval_freq', 10000,
    ]
)

run = exp.submit(config)
