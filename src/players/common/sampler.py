import numpy as np
import torch
import torch.nn.functional as f


def epsilon_soft_sample(action_dist, action_mask, epsilon=1) -> int:
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


def epsilon_greedy_sample(action_dist, action_mask, epsilon=0) -> int:
    if np.random.random() < epsilon:
        # unif random
        val_action_idxs = np.flatnonzero(action_mask)
        return np.random.choice(val_action_idxs)
    else:
        # greedy
        subset_idx = np.argmax(action_dist[action_mask])
        parent_idx = np.arange(action_dist.shape[0])[action_mask][subset_idx]
        return parent_idx


def epsilon_soft_sample_nd(action_dist, action_mask, epsilon=0) -> int:
    # assumes draw is the last idx
    val_action_idxs = np.flatnonzero(action_mask)
    if val_action_idxs.size > 1:
        action_dist = action_dist[:-1]
        action_mask = action_mask[:-1]

    return epsilon_soft_sample(action_dist, action_mask, epsilon)


def epsilon_greedy_sample_nd(action_dist, action_mask, epsilon=0) -> int:
    # assumes draw is the last idx
    val_action_idxs = np.flatnonzero(action_mask)
    if val_action_idxs.size > 1:
        action_dist = action_dist[:-1]
        action_mask = action_mask[:-1]

    return epsilon_greedy_sample(action_dist, action_mask, epsilon)


def epsilon_from_dist_sample(action_dist, action_mask, epsilon=1):
    if np.random.random() < epsilon:
        # from dist
        val_action_idxs = np.flatnonzero(action_mask)
        return np.random.choice(val_action_idxs, p=action_dist[action_mask])

    else:
        # greedy
        subset_idx = np.argmax(action_dist[action_mask])
        parent_idx = np.arange(action_dist.shape[0])[action_mask][subset_idx]
        return parent_idx


def epsilon_from_dist_sample_nd(action_dist, action_mask, epsilon=1) -> int:
    # assumes draw is the last idx
    val_action_idxs = np.flatnonzero(action_mask)
    if val_action_idxs.size > 1:
        action_dist = action_dist[:-1]
        action_mask = action_mask[:-1]

    return epsilon_from_dist_sample(action_dist, action_mask, epsilon)


def from_sampler_str(sampler_str):
    if sampler_str == "egnd":
        return epsilon_greedy_sample_nd
    elif sampler_str == "esnd":
        return epsilon_soft_sample_nd
    elif sampler_str == "es":
        return epsilon_soft_sample
    elif sampler_str == "eg":
        return epsilon_greedy_sample
    elif sampler_str == "efd":
        return epsilon_from_dist_sample
    elif sampler_str == "efdnd":
        return epsilon_from_dist_sample_nd
    else:
        print("Warning: Not using a valid sampler str")
        return epsilon_greedy_sample
