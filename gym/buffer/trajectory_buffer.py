import numpy as np
from collections import namedtuple
import random
import torch

experience_traj = namedtuple("trajectory", ["observations", "actions", "rewards", "summed_rewards", "traj_len"])

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class ReplayBuffer():
    def __init__(self, max_size, state_dim, action_dim, batch_size, max_len, max_ep_len=1000, device="cpu", rtg_scale=1.0, state_mean=0.0, state_std=1.0):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = device
        self.rtg_scale = rtg_scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.buffer = []
        
    @staticmethod
    def compute_reward2go(rewards):
        return np.cumsum(rewards)[::-1]

    def add_sample(self, states, actions, rewards):
        # r2g = self.compute_reward2go(rewards)
        # stacked numpy arrays (traj length, feature dim)
        traj = experience_traj(observations=states,
                               actions=actions,
                               rewards=rewards,
                               # r2g=r2g,
                               summed_rewards=sum(rewards),
                               traj_len=len(rewards))
        # insert left on first position
        self.buffer.insert(0, traj)
        # keep the max buffer size - cutting off oldest element
        self.buffer = self.buffer[:self.max_size]
    
    def sort(self):
        #sort buffer
        self.buffer = sorted(self.buffer, key = lambda i: i.summed_rewards, reverse=True)
    
    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch
    
    def get_nbest(self, n):
        self.sort()
        return self.buffer[:n]
    
    def __len__(self):
        return len(self.buffer)
    
    def get_time_step_dist(self,):
        traj_lengths = np.stack([traj.traj_len for traj in self.buffer])
        probs = traj_lengths / sum(traj_lengths)
        return probs
    
    def get_batch(self, ):
        # samples trajectories based on timesteps ~ trajectory length
        probs = self.get_time_step_dist()
        batch_inds = np.random.choice(
            np.arange(len(self.buffer)),
            size=self.batch_size,
            replace=True,
            p=probs,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(self.batch_size):
            traj = self.buffer[int(batch_inds[i])]
            si = random.randint(0, traj.rewards.shape[0] - 1)

            # get sequences from dataset
            s.append(traj.observations[si:si + self.max_len].reshape(1, -1, self.state_dim))
            a.append(traj.actions[si:si + self.max_len].reshape(1, -1, self.action_dim))
            r.append(traj.rewards[si:si + self.max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj.rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, self.max_len - tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.rtg_scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)

        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, rtg, timesteps, mask
