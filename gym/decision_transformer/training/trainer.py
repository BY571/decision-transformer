import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()


class OnlineTrainer:

    def __init__(self, model, stochastic, optimizer, batch_size, buffer, prefill, scheduler=None, collect_fn=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.buffer = buffer
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.collect_fn = collect_fn
        self.diagnostics = dict()
        self.target_reward = 1.0
        self.env_interaction_steps = 0
        self.prefill_buffer()
        self.stochastic = stochastic
        if self.stochastic:
            log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=model.device)
            multiplier_optimizer = torch.optim.AdamW(
                [log_entropy_multiplier],
                lr=1e-4,
                weight_decay=1e-4,
            )
            multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
                multiplier_optimizer,
                lambda steps: min((steps+1)/10000, 1)
            )
            loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(log_entropy_multiplier.detach()) * torch.mean(entropies)
            target_entropy = -model.act_dim
            entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (torch.mean(entropies.detach()) - target_entropy)
        else:
            self.loss_fn = torch.nn.MSELoss()
            
        self.start_time = time.time()

    def prefill_buffer(self, prefill_eps=10):
        achieved_rewards = []
        for i in range(prefill_eps):
            random_target = np.random.random() * 100
            trgt, steps = self.collect_samples(random_target)
            self.env_interaction_steps += steps
            achieved_rewards.append(trgt) 
        best_trgt = max(achieved_rewards)
        if best_trgt > self.target_reward:
            self.target_reward = best_trgt
            
    def collect_samples(self, target):
        state_history, action_history, reward_history = self.collect_fn(self.model, target)
        states, actions, rewards = np.stack(state_history), np.stack(action_history), np.stack(reward_history)
        self.buffer.add_sample(states=states, actions=actions, rewards=rewards)
        return sum(rewards), len(rewards)

    def train_iteration(self, num_steps=1, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        # add new data to the buffer
        self.model.eval()
        collect_return, traj_length = self.collect_samples(self.target_reward) # Maybe add noise?
        if collect_return > self.target_reward:
            self.target_reward = collect_return
        self.env_interaction_steps += traj_length
        logs['collect/reward'] = collect_return
        logs['collect/target_reward'] = self.target_reward
        logs['collect/trajectory_lengths'] = traj_length
        logs['collect/total_env_interactions'] = self.env_interaction_steps
        logs['collect/buffer_size'] = self.buffer.__len__()
        train_start = time.time()
        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        # eval_start = time.time()

        # self.model.eval()
        # for eval_fn in self.eval_fns:
        #     outputs = eval_fn(self.model)
        #     for k, v in outputs.items():
        #         logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        # logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, rtg, timesteps, attention_mask = self.buffer.get_batch()
        action_target = torch.clone(actions)
        rtg_target = torch.clone(rtg[:,:-1])
        
        state_preds, action_preds, reward_preds, _, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, target_actions=action_target
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        if self.stochastic:
            pass
        else:
            loss = self.loss_fn(action_preds, action_target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()