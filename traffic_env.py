"""
CityFlow environment wrapper for CoLight-style MARL traffic signal control.
Jinan 3x4 grid, 12 controllable intersections.
"""

import cityflow
import json
import numpy as np
import os


# ── Absolute paths ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_THIS_DIR, "colight", "data", "Jinan", "3_4")
ROADNET_FILE = "roadnet_3_4.json"
FLOW_FILE    = "anon_3_4_jinan_real.json"


def _write_cityflow_config(data_dir, roadnet_file, flow_file, seed=0):
    cfg = {
        "interval":       1.0,
        "seed":           seed,
        "dir":            data_dir + "/",
        "roadnetFile":    roadnet_file,
        "flowFile":       flow_file,
        "rlTrafficLight": True,
        "saveReplay":     False,
        "roadnetLogFile": "",
        "replayLogFile":  ""
    }
    path = "/tmp/cityflow_exp.json"
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


class TSCEnv:
    def __init__(self,
                 data_dir=DATA_DIR,
                 roadnet_file=ROADNET_FILE,
                 flow_file=FLOW_FILE,
                 num_steps=3600,
                 yellow_time=5,
                 seed=0):

        self.data_dir     = data_dir
        self.roadnet_file = roadnet_file
        self.flow_file    = flow_file
        self.num_steps    = num_steps
        self.yellow_time  = yellow_time
        self.seed         = seed

        with open(os.path.join(data_dir, roadnet_file)) as f:
            self.roadnet = json.load(f)

        # ── Controllable intersections ────────────────────────────────────
        self.intersections   = []
        self.inter_id_to_idx = {}
        for inter in self.roadnet["intersections"]:
            lp = inter["trafficLight"]["lightphases"]
            if any(len(p["availableRoadLinks"]) > 0 for p in lp):
                idx = len(self.intersections)
                self.inter_id_to_idx[inter["id"]] = idx
                self.intersections.append(inter)

        self.num_intersections = len(self.intersections)
        print(f"Controllable intersections: {self.num_intersections}")

        # ── Valid phases per intersection ─────────────────────────────────
        # Only keep phases that have available road links (skip all-red phases)
        # Cap at 4 phases to keep action space manageable
        self.phases = []
        for inter in self.intersections:
            lp    = inter["trafficLight"]["lightphases"]
            valid = [i for i, p in enumerate(lp)
                     if len(p["availableRoadLinks"]) > 0]
            if len(valid) < 2:
                valid = list(range(len(lp)))
            # Cap at 4 to keep action space small and stable
            valid = valid[:4]
            self.phases.append(valid)

        self.num_phases = [len(p) for p in self.phases]
        self.max_phases = max(self.num_phases)

        # ── Incoming lanes per intersection ───────────────────────────────
        self.inter_incoming_lanes = []
        for inter in self.intersections:
            incoming = []
            for road in self.roadnet.get("roads", []):
                if road.get("endIntersection") == inter["id"]:
                    num_lanes = len(road.get("lanes", []))
                    for ln in range(num_lanes):
                        incoming.append(f"{road['id']}_{ln}")
            self.inter_incoming_lanes.append(incoming)

        # ── Feature dim ───────────────────────────────────────────────────
        # Use up to 4 incoming road counts (N/S/E/W directions)
        self.phase_dim   = 8   # one-hot phase
        self.vehicle_dim = 4   # per-direction vehicle counts (up to 4 roads)
        self.feature_dim = self.phase_dim + self.vehicle_dim  # 12

        # ── Adjacency ─────────────────────────────────────────────────────
        self.TOP_K      = 5
        self.adj_matrix = self._build_adjacency()

        # ── Internal state ────────────────────────────────────────────────
        self.eng          = None
        self.cur_phases   = np.zeros(self.num_intersections, dtype=int)
        self.time_phase   = np.zeros(self.num_intersections, dtype=int)
        self.in_yellow    = np.zeros(self.num_intersections, dtype=bool)
        self.next_phase   = np.zeros(self.num_intersections, dtype=int)
        self.current_step = 0

    def _build_adjacency(self):
        positions = {}
        for inter in self.roadnet["intersections"]:
            if inter["id"] in self.inter_id_to_idx:
                idx = self.inter_id_to_idx[inter["id"]]
                positions[idx] = (inter["point"]["x"], inter["point"]["y"])

        N   = self.num_intersections
        adj = np.zeros((N, self.TOP_K), dtype=int)
        for i in range(N):
            xi, yi = positions[i]
            dists  = sorted(
                ((((xi-positions[j][0])**2+(yi-positions[j][1])**2)**0.5), j)
                for j in range(N)
            )
            nb = [j for _, j in dists[:self.TOP_K]]
            while len(nb) < self.TOP_K:
                nb.append(nb[-1])
            adj[i] = nb
        return adj

    def reset(self):
        cfg = _write_cityflow_config(
            self.data_dir, self.roadnet_file, self.flow_file, self.seed)
        self.eng = cityflow.Engine(cfg, thread_num=4)

        self.current_step = 0
        self.cur_phases   = np.zeros(self.num_intersections, dtype=int)
        self.time_phase   = np.zeros(self.num_intersections, dtype=int)
        self.in_yellow    = np.zeros(self.num_intersections, dtype=bool)
        self.next_phase   = np.zeros(self.num_intersections, dtype=int)

        for i, inter in enumerate(self.intersections):
            self.eng.set_tl_phase(inter["id"], self.phases[i][0])
        self.eng.next_step()
        return self._get_state()

    def step(self, actions):
        for i, inter in enumerate(self.intersections):
            action  = int(actions[i]) % self.num_phases[i]
            desired = self.phases[i][action]

            if self.in_yellow[i]:
                self.time_phase[i] += 1
                if self.time_phase[i] >= self.yellow_time:
                    self.eng.set_tl_phase(inter["id"], self.next_phase[i])
                    self.cur_phases[i] = self.next_phase[i]
                    self.in_yellow[i]  = False
                    self.time_phase[i] = 0
            else:
                if desired != self.cur_phases[i]:
                    self.eng.set_tl_phase(inter["id"], 0)
                    self.in_yellow[i]  = True
                    self.next_phase[i] = desired
                    self.time_phase[i] = 0

        self.eng.next_step()
        self.current_step += 1

        reward  = self._get_reward()
        rewards = np.full(self.num_intersections, reward, dtype=np.float32)
        done    = self.current_step >= self.num_steps
        info    = {"step": self.current_step,
                   "avg_travel_time": self.get_avg_travel_time()}
        return self._get_state(), rewards, done, info

    def _get_state(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        features    = np.zeros((self.num_intersections, self.feature_dim),
                               dtype=np.float32)
        for i in range(self.num_intersections):
            # Phase one-hot
            p_idx = self.cur_phases[i] % self.phase_dim
            features[i, p_idx] = 1.0

            # Per-road incoming vehicle counts (group lanes by road)
            # Find unique roads for this intersection
            seen_roads = {}
            for ln in self.inter_incoming_lanes[i]:
                # lane id format: road_x_y_z_lanenum -> road prefix is everything except last _N
                parts = ln.rsplit('_', 1)
                road_id = parts[0] if len(parts) == 2 else ln
                if road_id not in seen_roads:
                    seen_roads[road_id] = 0
                seen_roads[road_id] += lane_counts.get(ln, 0)

            road_counts = list(seen_roads.values())[:self.vehicle_dim]
            for k, cnt in enumerate(road_counts):
                features[i, self.phase_dim + k] = cnt / 20.0  # normalise
        return {"features": features,
                "adj":      self.adj_matrix.astype(np.int64)}

    def _get_reward(self):
        waiting = self.eng.get_lane_waiting_vehicle_count()
        total, count = 0, 0
        for lanes in self.inter_incoming_lanes:
            for ln in lanes:
                total += waiting.get(ln, 0)
                count += 1
        return -(total / max(count, 1))

    def get_avg_travel_time(self):
        try:
            return self.eng.get_average_travel_time()
        except Exception:
            return float("nan")