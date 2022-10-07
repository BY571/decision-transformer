
import atexit
import torch
import random
from torch import nn, Tensor
import numpy as np
import gym
from tqdm import tqdm
import os
import sys
import pickle

from decision_transformer.predictors.utils import get_transformer
from decision_transformer.predictors.decision_transformer import DTPredictor, StochDTPredictor
from decision_transformer.algorithm.decision_transformer import DecisionTransformer, OnlineDecisionTransformer
from buffer.trajectory_buffer import ReplayBuffer


def get_env(env_name):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
    else:
        raise NotImplementedError
    return env, env.observation_space.shape[0], env.action_space.shape[0]

def get_predictor(state_dim, act_dim, config):
    transformer = get_transformer(name=config.algorithm.predictor.name,
                                    embedding_dim=config.algorithm.predictor.hidden_size,
                                    n_layer=config.algorithm.predictor.n_layer,
                                    n_head=config.algorithm.predictor.n_head,
                                    n_inner= config.algorithm.predictor.n_inner,
                                    activation_function=config.algorithm.predictor.activation_function,
                                    n_positions=config.algorithm.predictor.n_positions,
                                    resid_pdrop=config.algorithm.predictor.resid_pdrop,
                                    attn_pdrop= config.algorithm.predictor.attn_pdrop)
    if config.algorithm.predictor.name == "gpt2":
        return DTPredictor(state_dim=state_dim,
                           act_dim=act_dim,
                           transformer=transformer,
                           hidden_size=config.algorithm.predictor.hidden_size,
                           max_length=config.eval_context,
                           max_ep_len=config.env.max_ep_len,
                           action_tanh=config.algorithm.predictor.action_tanh)
    elif config.algorithm.predictor.name == "stochastic_gpt2":
        return StochDTPredictor(state_dim,
                                act_dim=act_dim,
                                transformer=transformer,
                                hidden_size=config.algorithm.predictor.hidden_size,
                                max_length=config.eval_context,
                                max_ep_len=config.env.max_ep_len,
                                action_tanh=config.algorithm.predictor.action_tanh,
                                log_std_min=config.algorithm.predictor.log_std_min,
                                log_std_max=config.algorithm.predictor.log_std_max,
                                remove_pos_embs=config.algorithm.predictor.remove_pos_embs,
                                stochastic_tanh=config.algorithm.predictor.stochastic_tanh,
                                approximate_entropy_samples=config.algorithm.predictor.approximate_entropy_samples)
    else: 
        raise NotImplementedError


def get_algorithm(state_dim, act_dim, config):
    if config.algorithm.name == "dt":
        return DecisionTransformer(get_predictor(state_dim, act_dim, config), config)
    elif config.algorithm.name == "online-dt":
        return OnlineDecisionTransformer(predictor=get_predictor(state_dim, act_dim, config), config=config)
    else:
        raise NotImplementedError

def get_buffer(state_dim, act_dim, config):
    if config.buffer.name == "rtgbuffer":
        return ReplayBuffer(max_size=config.buffer.max_size,
                          state_dim=state_dim,
                          action_dim=act_dim,
                          batch_size=config.batch_size,
                          max_len=config.algorithm.predictor.K,
                          max_ep_len=config.env.max_ep_len,
                          device=config.device,
                          rtg_scale=config.buffer.scale,
                          state_mean=config.env.state_mean,
                          state_std=config.env.state_std)
    else:
        raise NotImplementedError

class Trainer():
    def __init__(self, config):
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        
        self.print_logs = config.print_logs
        self.epochs = config.epochs
        self.device = config.device
        self.max_ep_len = config.env.max_ep_len
        self.num_updates = config.num_updates
        self.max_interactions = config.max_interactions
        
        self.eval_runs = config.eval_runs
        
        self.state_mean = np.array(config.env.state_mean)
        self.state_std = np.array(config.env.state_std)
        
        # get_env
        self.env, self.state_dim, self.act_dim = get_env(config.env.name)
        self.env.seed(config.seed)
        # get_algorithm
        self.algo = get_algorithm(self.state_dim, self.act_dim, config)
        # get_data_source
        self.buffer = get_buffer(self.state_dim, self.act_dim, config)
        
        self.target_return = config.init_target_return
        
        # tracking values
        self.train_losses = []
        self.env_interaction_steps = 0
        self.update_steps = 0
        
        # prefill buffer
        if config.prefill_offline_data:
            self.load_offline_data2buffer(config)
        else:
            self.prefill_buffer(config.prefill_episodes)
        
    def load_offline_data2buffer(self, config):
        # get current working directory
        cwg = os.getcwd()
        root_path, _ = cwg.split("outputs") # split at outputs as we are currently in outputs 
        # load dataset
        dataset_path = f'data/{config.env.name}-{config.env.dataset}-v2.pkl'
        # combine paths
        dataset_path = root_path + dataset_path
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        # sort trajectories from worst to best
        returns = []
        for path in trajectories:
            returns.append(path['rewards'].sum())
        returns = np.array(returns)
        sorted_inds = np.argsort(returns)  # lowest to highest
        # select best x based on buffer size
        trajectories = [trajectories[index] for index in sorted_inds]
        trajectories = trajectories[config.buffer.max_size:]
        # add to buffer
        for traj in trajectories:
            self.buffer.add_sample(states=traj["observations"], actions=traj["actions"], rewards=traj["rewards"])
        
        print(f"Buffer filled with {self.buffer.__len__()} expert trajectories!")
        
    def prefill_buffer(self, prefill_episodes=10):
        achieved_rewards = []
        for i in range(prefill_episodes):
            random_target = np.random.random() * 100
            trgt, steps = self.collect_samples(random_target)
            self.env_interaction_steps += steps
            achieved_rewards.append(trgt) 
        best_trgt = max(achieved_rewards)
        if best_trgt > self.target_return:
            self.target_return = best_trgt
            
    def collect_samples(self, target_return):
        self.algo.set_eval_mode()
        state_history, action_history, reward_history = self.gather_episode(target_return)
        states, actions, rewards = np.stack(state_history), np.stack(action_history), np.stack(reward_history)
        self.buffer.add_sample(states=states, actions=actions, rewards=rewards)
        return sum(rewards), len(rewards)
    
    def evaluate(self, target_return):
        returns = []
        traj_len = []
        self.algo.set_eval_mode()
        for i in range(self.eval_runs):
            _, _, reward_history = self.gather_episode(target_return, eval=True)
            returns_ = np.stack(reward_history)
            returns.append(np.sum(returns_))
            traj_len.append(len(returns_))

        return {"evaluate/return_mean": np.mean(returns),
                "evaluate/return_std": np.std(returns),
                "evaluate/traj_len_mean": np.mean(traj_len),
                "evaluate/traj_len_std": np.std(traj_len)}
    
    def gather_episode(self, target_return, eval=False):

        # TODO: Use standard scaler? for now default values from each env
        state_mean = torch.from_numpy(self.state_mean).to(self.device)
        state_std = torch.from_numpy(self.state_std).to(self.device)

        state = self.env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=self.device, dtype=torch.float32)
        actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        state_history, action_history, reward_history = [], [], []
        
        for t in range(self.max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            action = self.algo.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                eval=eval,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            next_state, reward, done, _ = self.env.step(action)

            # add to history
            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)
            state = next_state
            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward


            pred_return = target_return[0,-1]
            target_return = torch.cat([target_return,
                                    pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps,
                                torch.ones((1, 1),
                                            device=self.device, dtype=torch.long) * (t+1)], dim=1)

            if done:
                break

        return state_history, action_history, reward_history
    
    def update(self, ):
        self.algo.set_train_mode()

        for _ in tqdm(range(1, self.num_updates+1), desc=f"Updates: {self.update_steps} - {self.update_steps + self.num_updates}", file=sys.stdout, leave=False):
            states, actions, rewards, rtg, timesteps, attention_mask = self.buffer.get_batch()
            loss_info = self.algo.train(states, actions, rewards, rtg, timesteps, attention_mask)
            self.update_steps += 1
            self.train_losses.append(loss_info["training/loss"])
        return loss_info

    def update_target_return(self, achieved_return):
        if achieved_return > self.target_return:
            self.target_return = achieved_return
    
    def train(self, wandb):
        wandb.watch(self.algo.predictor)
        for ep in tqdm(range(1, self.epochs+1), desc="Training", file=sys.stdout):
            
            # Gather new experience
            return_, steps = self.collect_samples(target_return=self.target_return)
            self.env_interaction_steps += steps
            self.update_target_return(return_)

            # Train Transformer
            loss_info = self.update()
            
            # Evaluate
            eval_info = self.evaluate(target_return=self.target_return)
            loss_info.update(eval_info)
            
            log_info = {"collect/reward": return_,
                        "collect/steps": self.env_interaction_steps,
                        "collect/trajectory_lengths": steps,
                        "collect/target_reward": self.target_return,
                        "collect/buffer_size": self.buffer.__len__(),
                        "training/loss_mean": np.mean(self.train_losses),
                        "training/loss_std": np.std(self.train_losses),
                        "training/updates": self.num_updates,
                        "training/epoch": ep
                        }
            log_info.update(loss_info)
            wandb.log(log_info)
            if self.print_logs:
                print("\n" + '=' * 80)
                print(f'Iteration {ep}')
                for k, v in log_info.items():
                    print(f'{k}: {v}')
            
            if self.env_interaction_steps >= self.max_interactions:
                break
                    
        