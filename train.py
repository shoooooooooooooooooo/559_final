import numpy as np
import json, random, sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, "/media/volume/GISense/bowen/xiaoshu/i2s_noise/text/CityFlow")
import cityflow

class TrafficEnvironment:
    def __init__(self, config_path, num_steps=3600,
                 sensor_failure_rate=0.0, comm_failure_rate=0.0):
        self.eng = cityflow.Engine(config_path, thread_num=4)
        self.num_steps = num_steps
        self.sensor_failure_rate = sensor_failure_rate
        self.comm_failure_rate   = comm_failure_rate
        self.current_step = 0
        config  = json.load(open(config_path))
        roadnet = json.load(open(config["dir"] + config["roadnetFile"]))
        self.intersections  = [i for i in roadnet["intersections"]
                               if not i.get("virtual", False)]
        self.num_agents     = len(self.intersections)
        self.agents_lanes   = self._get_agents_lanes(roadnet)
        self.adj_matrix     = self._build_adjacency(roadnet)
        self.num_phases     = [len(i["trafficLight"]["lightphases"])
                               for i in self.intersections]
        self.current_phases = [0] * self.num_agents
        # 记录实际 state_dim
        test_state = self._get_single_state(0, {})
        self.state_dim = len(test_state)
        print(f"环境初始化：{self.num_agents}个路口，state_dim={self.state_dim}")

    def _get_agents_lanes(self, roadnet):
        road_lanes = {r["id"]: [f"{r['id']}_{i}"
                      for i in range(len(r["lanes"]))]
                      for r in roadnet["roads"]}
        return [[l for rid in inter["roads"]
                   for l in road_lanes.get(rid, [])]
                for inter in self.intersections]

    def _build_adjacency(self, roadnet):
        ids    = [i["id"] for i in self.intersections]
        id2idx = {id_: i for i, id_ in enumerate(ids)}
        adj    = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        for r in roadnet["roads"]:
            s, e = r["startIntersection"], r["endIntersection"]
            if s in id2idx and e in id2idx:
                i, j = id2idx[s], id2idx[e]
                adj[i][j] = adj[j][i] = 1.0
        return adj

    def _inject_sensor(self, counts):
        if self.sensor_failure_rate == 0: return counts
        result = counts.copy()
        k = max(1, int(len(result) * self.sensor_failure_rate))
        for idx in random.sample(range(len(result)), min(k, len(result))):
            result[idx] = 0.0
        return result

    def _inject_comm(self, adj):
        if self.comm_failure_rate == 0: return adj
        result = adj.copy()
        edges  = [(i,j) for i in range(self.num_agents)
                        for j in range(i+1, self.num_agents)
                        if result[i][j] == 1]
        k = max(1, int(len(edges) * self.comm_failure_rate))
        for i, j in random.sample(edges, min(k, len(edges))):
            result[i][j] = result[j][i] = 0.0
        return result

    def _get_single_state(self, idx, raw):
        inter = self.intersections[idx]
        lanes = self.agents_lanes[idx]
        counts = self._inject_sensor([raw.get(l, 0) for l in lanes])
        phase  = [0] * self.num_phases[idx]
        phase[self.current_phases[idx]] = 1
        return np.array(phase + counts, dtype=np.float32)

    def get_state(self):
        raw = self.eng.get_lane_vehicle_count()
        return [self._get_single_state(i, raw) for i in range(self.num_agents)]

    def get_adjacency(self):
        return self._inject_comm(self.adj_matrix)

    def step(self, actions):
        for i, inter in enumerate(self.intersections):
            self.eng.set_tl_phase(inter["id"], actions[i])
            self.current_phases[i] = actions[i]
        self.eng.next_step()
        self.current_step += 1
        waiting = self.eng.get_lane_waiting_vehicle_count()
        rewards = [-sum(waiting.get(l, 0) for l in lanes)
                   for lanes in self.agents_lanes]
        return self.get_state(), rewards, self.current_step >= self.num_steps

    def reset(self):
        self.eng.reset()
        self.current_step   = 0
        self.current_phases = [0] * self.num_agents
        return self.get_state()

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x, adj):
        n = x.size(0)
        h = self.W(x)
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        e = F.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        e = e.masked_fill(adj == 0, float("-inf"))
        return F.elu(torch.matmul(torch.nan_to_num(F.softmax(e, dim=-1), nan=0.0), h))


class CoLightNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gat1    = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.gat2    = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.q_head  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions))

    def forward(self, states, adj):
        h = self.encoder(states)
        h_graph = self.gat2(self.gat1(h, adj), adj)
        return self.q_head(torch.cat([h, h_graph], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(np.array(a) for a in args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)


class CoLightTrainer:
    def __init__(self, env, model, lr=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000):
        self.env    = env
        self.model  = model
        self.device = next(model.parameters()).device
        self.target = CoLightNet(
            model.encoder[0].in_features,
            model.q_head[-1].out_features).to(self.device)
        self.target.load_state_dict(model.state_dict())
        self.opt           = torch.optim.Adam(model.parameters(), lr=lr)
        self.buffer        = ReplayBuffer()
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps   = 0

    def select_actions(self, states, adj):
        self.epsilon = self.epsilon_end + (1 - self.epsilon_end) * \
                       np.exp(-self.total_steps / self.epsilon_decay)
        if random.random() < self.epsilon:
            return [random.randint(0, self.env.num_phases[i]-1)
                    for i in range(self.env.num_agents)]
        with torch.no_grad():
            s = torch.FloatTensor(np.array(states)).to(self.device)
            a = torch.FloatTensor(adj).to(self.device)
            return self.model(s, a).argmax(dim=-1).cpu().numpy().tolist()

    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size: return None
        batch = self.buffer.sample(batch_size)
        s, a, act, r, ns, na, d = zip(*batch)
        s   = torch.FloatTensor(np.array(s)).to(self.device)
        a   = torch.FloatTensor(np.array(a)).to(self.device)
        ns  = torch.FloatTensor(np.array(ns)).to(self.device)
        na  = torch.FloatTensor(np.array(na)).to(self.device)
        act = torch.LongTensor(np.array(act)).to(self.device)
        r   = torch.FloatTensor(np.array(r)).to(self.device)
        d   = torch.FloatTensor(np.array(d)).to(self.device)
        B   = s.size(0)
        q_curr = torch.stack([self.model(s[b], a[b]) for b in range(B)])
        with torch.no_grad():
            q_next = torch.stack([self.target(ns[b], na[b]) for b in range(B)])
        q_curr = q_curr.gather(2, act.unsqueeze(-1)).squeeze(-1)
        q_tgt  = r + self.gamma * q_next.max(-1)[0] * (1 - d.unsqueeze(1))
        loss = F.mse_loss(q_curr, q_tgt.detach())
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def run_episode(self, batch_size=32):
        states = self.env.reset()
        adj    = self.env.get_adjacency()
        losses, total_r = [], 0
        while True:
            actions = self.select_actions(states, adj)
            next_states, rewards, done = self.env.step(actions)
            next_adj = self.env.get_adjacency()
            self.buffer.push(states, adj, actions, rewards,
                             next_states, next_adj, float(done))
            loss = self.train_step(batch_size)
            if loss: losses.append(loss)
            self.total_steps += 1
            if self.total_steps % 200 == 0:
                self.target.load_state_dict(self.model.state_dict())
            total_r += sum(rewards)
            states, adj = next_states, next_adj
            if done: break
        return total_r, np.mean(losses) if losses else 0, \
               self.env.get_average_travel_time()


if __name__ == "__main__":
    CONFIG   = "/media/volume/GISense/bowen/xiaoshu/i2s_noise/text/CityFlow/colight/data/Jinan/3_4/config_jinan.json"
    EPISODES = 150
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    env      = TrafficEnvironment(CONFIG, num_steps=3600)
    model    = CoLightNet(state_dim=env.state_dim,
                          num_actions=max(env.num_phases)).to(device)
    trainer  = CoLightTrainer(env, model)

    os.makedirs("checkpoints_v2", exist_ok=True)
    best_att = float("inf")

    print(f"{'Ep':>4} {'行驶时间':>10} {'奖励':>12} {'损失':>8} {'Eps':>6}")
    print("-" * 50)

    for ep in range(1, EPISODES + 1):
        reward, loss, att = trainer.run_episode()
        print(f"{ep:>4} {att:>10.2f} {reward:>12.1f} {loss:>8.4f} {trainer.epsilon:>6.3f}",
              flush=True)
        if att < best_att:
            best_att = att
            torch.save(model.state_dict(), "checkpoints_v2/best_model.pt")
            # state_dim 也一起保存
            torch.save({"state_dim": env.state_dim,
                        "num_actions": max(env.num_phases),
                        "weights": model.state_dict()},
                       "checkpoints_v2/best_model_meta.pt")
        if ep % 20 == 0:
            print(f"  → 最佳行驶时间：{best_att:.2f}s", flush=True)

    print(f"\n训练完成！最佳平均行驶时间：{best_att:.2f}s")
    print(f"state_dim={env.state_dim}, num_actions={max(env.num_phases)}")
