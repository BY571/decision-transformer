
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage official code

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt -w True
```

Adding `-w True` will log results to Weights and Biases.

## Example usage updated code 

```
python run.py 
```
Define your run by `run_name=my_experiment`
Adding `mode=online` will log results to Weights and Biases.

