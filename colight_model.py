"""
CoLight: Graph Attention Network for Multi-Agent Traffic Signal Control
PyTorch implementation based on:
  "CoLight: Learning Network-level Cooperation for Traffic Signal Control"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Graph attention over neighbors.
    Input:
      agent_feat:    (B, N, d)       - encoded features for each intersection
      neighbor_feat: (B, N, K, d)    - encoded features of K neighbors
    Output:
      out:           (B, N, dout)    - attended representation
      att:           (B, N, nhead, K) - attention weights
    """
    def __init__(self, d_in, dv, dout, nhead):
        super().__init__()
        self.nhead = nhead
        self.dv = dv
        # Project agent and neighbor to query/key space
        self.W_agent = nn.Linear(d_in, dv * nhead)
        self.W_neighbor = nn.Linear(d_in, dv * nhead)
        # Value projection
        self.W_value = nn.Linear(d_in, dv * nhead)
        # Output projection
        self.W_out = nn.Linear(dv * nhead, dout)

    def forward(self, agent_feat, neighbor_feat):
        """
        agent_feat:    (B, N, d)
        neighbor_feat: (B, N, K, d)
        """
        B, N, d = agent_feat.shape
        K = neighbor_feat.shape[2]

        # Query from agent: (B, N, nhead, dv)
        Q = self.W_agent(agent_feat).view(B, N, self.nhead, self.dv)
        # Key from neighbors: (B, N, K, nhead, dv)
        K_proj = self.W_neighbor(neighbor_feat).view(B, N, K, self.nhead, self.dv)
        # Value from neighbors: (B, N, K, nhead, dv)
        V = self.W_value(neighbor_feat).view(B, N, K, self.nhead, self.dv)

        # Attention scores: Q (B,N,nhead,dv) x K (B,N,K,nhead,dv)
        # -> (B, N, nhead, K)
        Q_exp = Q.unsqueeze(3)  # (B, N, nhead, 1, dv)
        K_exp = K_proj.permute(0, 1, 3, 2, 4)  # (B, N, nhead, K, dv)
        scores = torch.matmul(Q_exp, K_exp.transpose(-1, -2))  # (B, N, nhead, 1, K)
        scores = scores.squeeze(-2) / (self.dv ** 0.5)  # (B, N, nhead, K)
        att = F.softmax(scores, dim=-1)  # (B, N, nhead, K)

        # Weighted sum of values
        # att: (B, N, nhead, K) -> (B, N, nhead, 1, K)
        att_exp = att.unsqueeze(-2)
        V_exp = V.permute(0, 1, 3, 2, 4)  # (B, N, nhead, K, dv)
        out = torch.matmul(att_exp, V_exp).squeeze(-2)  # (B, N, nhead, dv)
        out = out.reshape(B, N, self.nhead * self.dv)  # (B, N, nhead*dv)
        out = F.relu(self.W_out(out))  # (B, N, dout)

        return out, att


class CoLightNet(nn.Module):
    """
    CoLight Q-network with graph attention communication.

    Architecture:
      1. MLP encoder: feature_dim -> embed_dim
      2. Graph attention layers (1 or 2)
      3. Q-value head: -> num_actions
    """
    def __init__(self, feature_dim, num_actions, num_agents, top_k,
                 embed_dim=32, att_dim=16, out_dim=32, nhead=1, num_layers=1):
        super().__init__()
        self.num_agents = num_agents
        self.top_k = top_k
        self.num_actions = num_actions
        self.num_layers = num_layers

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        # Graph attention layers
        self.att_layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = embed_dim if i == 0 else out_dim
            self.att_layers.append(
                MultiHeadAttention(d_in=d_in, dv=att_dim, dout=out_dim, nhead=nhead)
            )

        # Q-value output
        self.q_head = nn.Linear(out_dim, num_actions)

    def forward(self, features, adj):
        """
        features: (B, N, feature_dim)  - node features
        adj:      (B, N, K)            - neighbor indices (int)

        Returns:
          q_values: (B, N, num_actions)
          att_list: list of (B, N, nhead, K) attention weights per layer
        """
        B, N, _ = features.shape

        # Encode features
        h = self.encoder(features)  # (B, N, embed_dim)

        att_list = []
        for layer in self.att_layers:
            # Gather neighbor features using adj indices
            # adj: (B, N, K) -> expand to (B, N, K, embed_dim)
            neighbor_h = self._gather_neighbors(h, adj)  # (B, N, K, d)
            h, att = layer(h, neighbor_h)
            att_list.append(att)

        q_values = self.q_head(h)  # (B, N, num_actions)
        return q_values, att_list

    def _gather_neighbors(self, h, adj):
        """
        h:   (B, N, d)
        adj: (B, N, K) or (N, K) - neighbor indices

        Returns: (B, N, K, d)
        """
        B, N, d = h.shape

        if adj.dim() == 2:
            # (N, K) -> (B, N, K)
            adj = adj.unsqueeze(0).expand(B, -1, -1)

        K = adj.shape[2]
        # Flatten adj to (B, N*K), gather from h (B, N, d)
        idx = adj.reshape(B, -1)  # (B, N*K)
        # Expand h for gathering: (B, N*K, d)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
        gathered = torch.gather(h.clone(), 1, idx_exp)  # (B, N*K, d)
        neighbor_h = gathered.reshape(B, N, K, d)
        return neighbor_h


class ReplayBuffer:
    """Simple replay buffer for DQN."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state_feat, state_adj, actions, rewards, next_feat, next_adj, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state_feat, state_adj, actions, rewards, next_feat, next_adj, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        sf, sa, ac, rw, nf, na, dn = zip(*batch)
        return (
            np.stack(sf),
            np.stack(sa),
            np.stack(ac),
            np.stack(rw),
            np.stack(nf),
            np.stack(na),
            np.array(dn)
        )

    def __len__(self):
        return len(self.buffer)


class CoLightAgent:
    """
    DQN agent wrapping CoLightNet.
    Handles action selection, training, target network updates.
    """
    def __init__(self, feature_dim, num_actions, num_agents, top_k,
                 lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.98, batch_size=64, update_target_freq=5,
                 embed_dim=64, att_dim=32, out_dim=64, nhead=2, num_layers=2,
                 device='cpu'):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.device = torch.device(device)
        self.train_step = 0

        self.q_net = CoLightNet(
            feature_dim, num_actions, num_agents, top_k,
            embed_dim, att_dim, out_dim, nhead, num_layers
        ).to(self.device)

        self.target_net = CoLightNet(
            feature_dim, num_actions, num_agents, top_k,
            embed_dim, att_dim, out_dim, nhead, num_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

    def select_actions(self, features, adj):
        """
        Epsilon-greedy action selection.
        features: np.array (N, feature_dim)
        adj:      np.array (N, K)
        Returns: np.array (N,) int
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions, size=self.num_agents)

        feat_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        adj_t = torch.LongTensor(adj).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q, _ = self.q_net(feat_t, adj_t)
        actions = q.squeeze(0).argmax(dim=-1).cpu().numpy()
        return actions

    def store(self, state, actions, rewards, next_state, done):
        self.replay.push(
            state['features'], state['adj'],
            actions, rewards,
            next_state['features'], next_state['adj'],
            done
        )

    def train(self):
        if len(self.replay) < self.batch_size:
            return None

        sf, sa, ac, rw, nf, na, dn = self.replay.sample(self.batch_size)

        sf_t = torch.FloatTensor(sf).to(self.device)   # (B, N, F)
        sa_t = torch.LongTensor(sa).to(self.device)    # (B, N, K)
        ac_t = torch.LongTensor(ac).to(self.device)    # (B, N)
        rw_t = torch.FloatTensor(rw).to(self.device)   # (B, N)
        nf_t = torch.FloatTensor(nf).to(self.device)
        na_t = torch.LongTensor(na).to(self.device)
        dn_t = torch.FloatTensor(dn).to(self.device)   # (B,)

        # Current Q values
        q_vals, _ = self.q_net(sf_t, sa_t)             # (B, N, A)
        q_taken = q_vals.gather(-1, ac_t.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Target Q values
        with torch.no_grad():
            q_next, _ = self.target_net(nf_t, na_t)    # (B, N, A)
            q_next_max = q_next.max(dim=-1).values      # (B, N)
            # done flag: (B,) -> (B, 1) broadcast
            targets = rw_t + self.gamma * q_next_max * (1 - dn_t.unsqueeze(-1))

        loss = F.smooth_l1_loss(q_taken, targets)  # Huber loss, more stable than MSE

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.epsilon_min  # eval mode