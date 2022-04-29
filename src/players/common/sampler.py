import numpy as np
import torch
import torch.nn.functional as f


def epsilon_soft_sample(action_dist, action_mask, epsilon=1):
    if np.random.random() < epsilon:
        # softmax
        val_action_idxs = np.flatnonzero(action_mask)

        soft_dist = f.softmax(torch.from_numpy(action_dist[action_mask]), -1).numpy()

        return np.random.choice(val_action_idxs, p=soft_dist)
    else:
        # greedy
        subset_idx = np.argmax(action_dist[action_mask])
        parent_idx = np.arange(action_dist.shape[0])[action_mask][subset_idx]
        return parent_idx


def epsilon_greedy_sample(action_dist, action_mask, epsilon=0):
    if np.random.random() < epsilon:
        # unif random
        val_action_idxs = np.flatnonzero(action_mask)
        return np.random.choice(val_action_idxs)
    else:
        # greedy
        subset_idx = np.argmax(action_dist[action_mask])
        parent_idx = np.arange(action_dist.shape[0])[action_mask][subset_idx]
        return parent_idx
