"""
Failure mode injection for CoLight diagnostic study.

Two main failure types:
  1. Sensor failure  -> node feature masking
  2. Communication failure -> graph edge dropping

Additional modes:
  3. Feature noise
  4. Delayed observation
"""

import numpy as np
import copy


def apply_node_masking(features, failure_rate, rng=None):
    """
    Sensor failure: randomly mask node features to zero.

    Args:
        features:     np.array (N, F) - node feature matrix
        failure_rate: float in [0, 1] - fraction of nodes to mask
        rng:          np.random.Generator or None

    Returns:
        masked_features: np.array (N, F)
        failed_nodes:    list of int - which nodes were masked
    """
    if rng is None:
        rng = np.random.default_rng()

    N = features.shape[0]
    num_failed = int(N * failure_rate)
    failed_nodes = rng.choice(N, size=num_failed, replace=False).tolist()

    masked = features.copy()
    for node in failed_nodes:
        masked[node, :] = 0.0  # complete sensor blackout

    return masked, failed_nodes


def apply_edge_dropping(adj, failure_rate, rng=None):
    """
    Communication failure: randomly drop graph edges.

    For each node, independently drop each neighbor connection with
    probability = failure_rate. Self-loop (adj[i][0] == i) is never dropped.

    Args:
        adj:          np.array (N, K) - neighbor index matrix
        failure_rate: float in [0, 1] - probability of dropping each edge
        rng:          np.random.Generator or None

    Returns:
        dropped_adj:   np.array (N, K) - modified adjacency
                       dropped edges are replaced with self-loop (node itself)
        dropped_edges: list of (i, k) tuples indicating dropped positions
    """
    if rng is None:
        rng = np.random.default_rng()

    N, K = adj.shape
    dropped_adj = adj.copy()
    dropped_edges = []

    for i in range(N):
        for k in range(K):
            neighbor = adj[i, k]
            if neighbor == i:
                continue  # never drop self-loop
            if rng.random() < failure_rate:
                # Replace with self (node i falls back to its own state)
                dropped_adj[i, k] = i
                dropped_edges.append((i, k))

    return dropped_adj, dropped_edges


def apply_feature_noise(features, noise_std=0.1, rng=None):
    """
    Sensor noise: add Gaussian noise to node features.

    Args:
        features:  np.array (N, F)
        noise_std: float - standard deviation of noise
        rng:       np.random.Generator or None

    Returns:
        noisy_features: np.array (N, F), clipped to [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, noise_std, size=features.shape).astype(np.float32)
    noisy = features + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


class DelayBuffer:
    """
    Observation delay: stores past observations and returns stale ones.
    """
    def __init__(self, delay_steps=2):
        self.delay = delay_steps
        self.history = []

    def push(self, features):
        """Push current observation, return delayed one."""
        self.history.append(features.copy())
        if len(self.history) > self.delay + 1:
            self.history.pop(0)

        if len(self.history) <= self.delay:
            # Not enough history yet: return zeros (no observation)
            return np.zeros_like(features)
        else:
            return self.history[0]  # return oldest in window

    def reset(self, features):
        """Reset buffer with initial observation."""
        self.history = [features.copy()] * (self.delay + 1)
        return features




def apply_spatial_block_masking(features, adj, failure_rate, rng=None):
    """
    Spatial block failure: mask a center node and all its neighbors.
    Models region-wide sensor outage (e.g. power failure in a district).

    Args:
        features:     np.array (N, F)
        adj:          np.array (N, K) - neighbor indices
        failure_rate: float - fraction of nodes to affect (determines block size)
        rng:          np.random.Generator or None

    Returns:
        masked_features: np.array (N, F)
        failed_nodes:    list of int
    """
    if rng is None:
        rng = np.random.default_rng()

    N = features.shape[0]
    masked = features.copy()
    failed_nodes = set()

    # Number of center nodes to pick
    num_centers = max(1, int(N * failure_rate / 3))
    centers = rng.choice(N, size=num_centers, replace=False)

    for center in centers:
        failed_nodes.add(int(center))
        # Also mask all neighbors of center
        for neighbor in adj[center]:
            failed_nodes.add(int(neighbor))

    for node in failed_nodes:
        masked[node, :] = 0.0

    return masked, list(failed_nodes)


def apply_high_degree_masking(features, adj, failure_rate, rng=None):
    """
    High-degree failure: preferentially mask nodes with most neighbors.
    Models failure of hub intersections (e.g. major junctions).

    Args:
        features:     np.array (N, F)
        adj:          np.array (N, K)
        failure_rate: float
        rng:          np.random.Generator or None

    Returns:
        masked_features: np.array (N, F)
        failed_nodes:    list of int
    """
    if rng is None:
        rng = np.random.default_rng()

    N = features.shape[0]
    num_failed = max(1, int(N * failure_rate))

    # Compute degree: count unique non-self neighbors
    degrees = []
    for i in range(N):
        unique_nb = set(adj[i].tolist()) - {i}
        degrees.append(len(unique_nb))

    # Sort by degree descending, take top num_failed
    sorted_nodes = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    failed_nodes = sorted_nodes[:num_failed]

    masked = features.copy()
    for node in failed_nodes:
        masked[node, :] = 0.0

    return masked, failed_nodes
class FailureInjector:
    """
    Central failure injection wrapper.
    Applied to state dict before passing to the agent.
    """
    def __init__(self, mode, failure_rate,
                 noise_std=0.1, delay_steps=2, seed=42):
        """
        mode: one of
          'clean'       - no failure
          'node_mask'   - sensor failure (node feature masking)
          'edge_drop'   - communication failure (edge dropping)
          'feat_noise'  - sensor noise
          'delay'       - observation delay
        failure_rate: float [0, 1] - severity
        """
        assert mode in ('clean', 'node_mask', 'edge_drop', 'feat_noise', 'delay',
                        'spatial_block', 'high_degree'), \
            f"Unknown mode: {mode}"
        self.mode = mode
        self.failure_rate = failure_rate
        self.noise_std = noise_std
        self.delay_buffer = DelayBuffer(delay_steps)
        self.rng = np.random.default_rng(seed)

    def inject(self, state):
        """
        Apply failure to state dict.
        state: dict with 'features' (N, F) and 'adj' (N, K)
        Returns: modified state dict (new copy, original unchanged)
        """
        features = state['features'].copy()
        adj = state['adj'].copy()

        if self.mode == 'clean':
            pass

        elif self.mode == 'node_mask':
            features, _ = apply_node_masking(features, self.failure_rate, self.rng)

        elif self.mode == 'edge_drop':
            adj, _ = apply_edge_dropping(adj, self.failure_rate, self.rng)

        elif self.mode == 'feat_noise':
            features = apply_feature_noise(features, self.noise_std, self.rng)

        elif self.mode == 'delay':
            features = self.delay_buffer.push(features)

        elif self.mode == 'spatial_block':
            features, _ = apply_spatial_block_masking(
                features, adj, self.failure_rate, self.rng)

        elif self.mode == 'high_degree':
            features, _ = apply_high_degree_masking(
                features, adj, self.failure_rate, self.rng)

        return {'features': features.astype(np.float32), 'adj': adj}

    def reset(self, state):
        """Call at the start of each episode."""
        if self.mode == 'delay':
            self.delay_buffer.reset(state['features'])


# ── Convenience functions for experiments ────────────────────────────────────

def make_injector(mode, failure_rate, seed=42):
    """Factory for FailureInjector."""
    return FailureInjector(mode=mode, failure_rate=failure_rate, seed=seed)


FAILURE_MODES = ['clean', 'node_mask', 'edge_drop', 'feat_noise', 'delay', 'spatial_block', 'high_degree']
FAILURE_RATES = [0.0, 0.1, 0.2, 0.3]