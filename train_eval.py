"""
Main experiment script for:
  "Failure Modes of Graph-Attention Traffic Signal Control:
   A Diagnostic Study under Sensor and Communication Failures"

Experiments:
  A) Failure-type sensitivity (clean model tested under each failure mode)
  B) Cross-failure robustness transfer (train on one, test on another)

Usage:
  python train_eval.py --mode train_clean
  python train_eval.py --mode eval_all
  python train_eval.py --mode cross_failure
  python train_eval.py --mode full   # runs everything
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from traffic_env import TSCEnv
from colight_model import CoLightAgent
from failure_modes import FailureInjector, FAILURE_MODES, FAILURE_RATES


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
MODEL_DIR   = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# Training hyperparameters (tuned for 1-2 day budget)
TRAIN_EPISODES  = 20      # episodes per training run (increase if time allows)
EVAL_EPISODES   = 3       # episodes per eval setting
STEPS_PER_EP    = 3600    # 1 hour of simulation
EVAL_STEPS      = 900     # 15-min eval episodes (faster)

AGENT_KWARGS = dict(
    lr=5e-4,
    gamma=0.95,
    epsilon=1.0,          # starts fully random
    epsilon_min=0.05,
    epsilon_decay=0.95,   # per-episode: 1.0 * 0.95^60 ~ 0.05, reaches min ~ep60
    batch_size=64,
    update_target_freq=5,
    embed_dim=64,
    att_dim=32,
    out_dim=64,
    nhead=2,
    num_layers=2,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(num_steps=STEPS_PER_EP):
    from traffic_env import DATA_DIR, ROADNET_FILE, FLOW_FILE
    env = TSCEnv(
        data_dir=DATA_DIR,
        roadnet_file=ROADNET_FILE,
        flow_file=FLOW_FILE,
        num_steps=num_steps
    )
    return env


# ── Training ──────────────────────────────────────────────────────────────────

def train(train_failure_mode='clean', train_failure_rate=0.2,
          num_episodes=TRAIN_EPISODES, save_name=None):
    """
    Train a CoLight agent, optionally with failure injection during training.

    train_failure_mode: 'clean', 'node_mask', 'edge_drop', etc.
    train_failure_rate: failure rate used during training (if not clean)
    """
    print(f"\n{'='*60}")
    print(f"Training | mode={train_failure_mode} rate={train_failure_rate}")
    print(f"{'='*60}")

    env = make_env(num_steps=STEPS_PER_EP)
    state = env.reset()

    agent = CoLightAgent(
        feature_dim=env.feature_dim,
        num_actions=max(env.num_phases),
        num_agents=env.num_intersections,
        top_k=env.TOP_K,
        **AGENT_KWARGS
    )

    injector = FailureInjector(
        mode=train_failure_mode,
        failure_rate=train_failure_rate if train_failure_mode != 'clean' else 0.0
    )

    episode_rewards = []
    episode_travel_times = []

    for ep in range(num_episodes):
        state = env.reset()
        injector.reset(state)
        ep_reward = 0
        losses = []

        for step in range(STEPS_PER_EP):
            # Inject failure into observed state
            obs = injector.inject(state)

            # Select actions
            actions = agent.select_actions(obs['features'], obs['adj'])

            # Step environment (no failure in actual env)
            next_state, rewards, done, info = env.step(actions)

            # Inject failure into next state for storing
            next_obs = injector.inject(next_state)

            # Normalize reward to [-1, 0] range for stable training
            norm_rewards = rewards / 20.0
            # Store transition
            agent.store(obs, actions, norm_rewards, next_obs, done)

            # Train
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            ep_reward += rewards.mean()
            state = next_state

            if done:
                break

        avg_travel_time = env.get_avg_travel_time()
        episode_rewards.append(ep_reward)
        episode_travel_times.append(avg_travel_time)
        avg_loss = np.mean(losses) if losses else 0

        # Decay epsilon once per episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.95)

        print(f"  Episode {ep+1:3d}/{num_episodes} | "
              f"reward={ep_reward/STEPS_PER_EP:.3f}/step | "
              f"travel_time={avg_travel_time:.1f}s | "
              f"loss={avg_loss:.4f} | "
              f"eps={agent.epsilon:.3f}")

    # Save model
    if save_name is None:
        save_name = f"colight_{train_failure_mode}"
    model_path = os.path.join(MODEL_DIR, f"{save_name}.pt")
    agent.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save training curve
    np.save(os.path.join(RESULTS_DIR, f"train_rewards_{save_name}.npy"),
            np.array(episode_rewards))
    np.save(os.path.join(RESULTS_DIR, f"train_travel_times_{save_name}.npy"),
            np.array(episode_travel_times))

    return agent, episode_travel_times


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, test_failure_mode, test_failure_rate,
             num_episodes=EVAL_EPISODES, seed=0):
    """
    Evaluate a trained agent under a specific failure mode and rate.
    Returns mean average travel time across episodes.
    Each episode uses a different seed so results vary.
    """
    travel_times = []
    device = next(agent.q_net.parameters()).device

    for ep in range(num_episodes):
        # Different seed per episode -> different traffic randomness
        ep_seed = seed + ep * 7
        env = make_env(num_steps=EVAL_STEPS)
        env.seed = ep_seed

        injector = FailureInjector(
            mode=test_failure_mode,
            failure_rate=test_failure_rate,
            seed=ep_seed + 1
        )

        state = env.reset()
        injector.reset(state)

        for step in range(EVAL_STEPS):
            obs = injector.inject(state)
            feat_t = torch.FloatTensor(obs['features']).unsqueeze(0).to(device)
            adj_t  = torch.LongTensor(obs['adj']).unsqueeze(0).to(device)
            with torch.no_grad():
                q, _ = agent.q_net(feat_t, adj_t)
            actions = q.squeeze(0).argmax(dim=-1).cpu().numpy()
            next_state, rewards, done, info = env.step(actions)
            state = next_state
            if done:
                break

        travel_times.append(env.get_avg_travel_time())

    mean_tt = np.mean(travel_times)
    std_tt  = np.std(travel_times)
    return mean_tt, std_tt


# ── Experiment A: Failure-type sensitivity ────────────────────────────────────

def experiment_A(agent, agent_name="clean"):
    """
    Test a single trained agent across all failure modes and rates.
    Returns results dict: {mode: {rate: (mean_tt, std_tt)}}
    """
    print(f"\n{'='*60}")
    print(f"Experiment A: Failure-type sensitivity | agent={agent_name}")
    print(f"{'='*60}")

    # Test modes for experiment A
    test_modes = ['clean', 'node_mask', 'edge_drop', 'feat_noise']
    results = defaultdict(dict)

    for mode in test_modes:
        for rate in FAILURE_RATES:
            if mode == 'clean' and rate > 0:
                continue  # clean has no rate
            actual_rate = rate if mode != 'clean' else 0.0
            mean_tt, std_tt = evaluate(agent, mode, actual_rate)
            results[mode][rate] = (mean_tt, std_tt)
            print(f"  mode={mode:12s} rate={rate:.1f} | "
                  f"travel_time={mean_tt:.1f} ± {std_tt:.1f}s")

    # Save
    save_path = os.path.join(RESULTS_DIR, f"expA_{agent_name}.json")
    json.dump({m: {str(r): v for r, v in rv.items()}
               for m, rv in results.items()}, open(save_path, 'w'), indent=2)
    print(f"Results saved to {save_path}")
    return results


# ── Experiment B: Cross-failure robustness transfer ───────────────────────────

def experiment_B(agents_dict):
    """
    3 x 2 cross-failure matrix.
    agents_dict: {'clean': agent, 'node_mask': agent, 'edge_drop': agent}
    Returns: nested dict [train_mode][test_mode][rate] = (mean, std)
    """
    print(f"\n{'='*60}")
    print(f"Experiment B: Cross-failure robustness transfer")
    print(f"{'='*60}")

    train_modes = ['clean', 'node_mask', 'edge_drop']
    test_modes  = ['node_mask', 'edge_drop']

    results = defaultdict(lambda: defaultdict(dict))

    for train_mode, agent in agents_dict.items():
        if train_mode not in train_modes:
            continue
        for test_mode in test_modes:
            for rate in FAILURE_RATES:
                mean_tt, std_tt = evaluate(agent, test_mode, rate)
                results[train_mode][test_mode][rate] = (mean_tt, std_tt)
                print(f"  train={train_mode:12s} test={test_mode:12s} "
                      f"rate={rate:.1f} | {mean_tt:.1f} ± {std_tt:.1f}s")

    save_path = os.path.join(RESULTS_DIR, "expB_cross_failure.json")
    json.dump(
        {tm: {tt: {str(r): v for r, v in rv.items()}
              for tt, rv in tv.items()}
         for tm, tv in results.items()},
        open(save_path, 'w'), indent=2
    )
    print(f"Results saved to {save_path}")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_experiment_A(results, agent_name="clean"):
    """Line plot: travel time vs failure rate for each failure mode."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mode_styles = {
        'node_mask':  ('tab:red',    '-o',  'Sensor Failure (node mask)'),
        'edge_drop':  ('tab:blue',   '-s',  'Comm. Failure (edge drop)'),
        'feat_noise': ('tab:orange', '-^',  'Feature Noise'),
    }

    # Baseline (clean, rate=0)
    baseline_tt = results['clean'][0.0][0]
    ax.axhline(baseline_tt, color='gray', linestyle='--', label=f'Clean baseline ({baseline_tt:.0f}s)')

    for mode, (color, marker, label) in mode_styles.items():
        if mode not in results:
            continue
        rates = sorted(results[mode].keys())
        means = [results[mode][r][0] for r in rates]
        stds  = [results[mode][r][1] for r in rates]
        ax.plot(rates, means, marker, color=color, label=label, linewidth=2)
        ax.fill_between(rates,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=color)

    ax.set_xlabel('Failure Rate', fontsize=13)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=13)
    ax.set_title(f'Failure Type Sensitivity (trained: {agent_name})', fontsize=14)
    ax.set_xticks(FAILURE_RATES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULTS_DIR, f"expA_{agent_name}_sensitivity.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")


def plot_experiment_B(results):
    """
    Two subplots:
      Left:  test on node_mask, compare train={clean, node_mask, edge_drop}
      Right: test on edge_drop, compare train={clean, node_mask, edge_drop}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    test_modes = ['node_mask', 'edge_drop']
    test_labels = ['Sensor Failure (node mask)', 'Comm. Failure (edge drop)']

    train_styles = {
        'clean':     ('gray',      '-o',  'Clean-trained'),
        'node_mask': ('tab:red',   '-s',  'Sensor-trained'),
        'edge_drop': ('tab:blue',  '-^',  'Comm-trained'),
    }

    for ax, test_mode, test_label in zip(axes, test_modes, test_labels):
        for train_mode, (color, marker, label) in train_styles.items():
            if train_mode not in results or test_mode not in results[train_mode]:
                continue
            rv = results[train_mode][test_mode]
            rates = sorted(rv.keys())
            means = [rv[r][0] for r in rates]
            stds  = [rv[r][1] for r in rates]
            ax.plot(rates, means, marker, color=color, label=label, linewidth=2)
            ax.fill_between(rates,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)

        ax.set_xlabel('Failure Rate', fontsize=12)
        ax.set_ylabel('Avg. Travel Time (s)', fontsize=12)
        ax.set_title(f'Test: {test_label}', fontsize=13)
        ax.set_xticks(FAILURE_RATES)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-Failure Robustness Transfer', fontsize=15)
    path = os.path.join(RESULTS_DIR, "expB_cross_failure.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")


def plot_training_curves(names):
    """Plot training travel time curves for all trained agents."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['gray', 'tab:red', 'tab:blue']
    for name, color in zip(names, colors):
        path = os.path.join(RESULTS_DIR, f"train_travel_times_{name}.npy")
        if not os.path.exists(path):
            continue
        tt = np.load(path)
        ax.plot(tt, color=color, label=name, linewidth=2)

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=13)
    ax.set_title('Training Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curve saved.")


# ── Main ──────────────────────────────────────────────────────────────────────



# ── Experiment C: Failure pattern analysis ────────────────────────────────────

def experiment_C(agent, agent_name="clean"):
    """
    Compare different failure patterns at fixed rate (20%).
    Tests: random node mask vs spatial block vs high degree vs random edge drop
    """
    print(f"\n{'='*60}")
    print(f"Experiment C: Failure pattern analysis | agent={agent_name}")
    print(f"{'='*60}")

    FIXED_RATE = 0.2
    patterns = [
        ('node_mask',     FIXED_RATE, 'Random Sensor Failure'),
        ('spatial_block', FIXED_RATE, 'Spatial Block Failure'),
        ('high_degree',   FIXED_RATE, 'High-Degree Failure'),
        ('edge_drop',     FIXED_RATE, 'Random Comm. Failure'),
        ('delay',         FIXED_RATE, 'Delayed Observation'),
    ]

    results = {}
    for mode, rate, label in patterns:
        mean_tt, std_tt = evaluate(agent, mode, rate)
        results[mode] = (mean_tt, std_tt, label)
        print(f"  {label:30s} | {mean_tt:.1f} ± {std_tt:.1f}s")

    # Save
    import json
    save_path = os.path.join(RESULTS_DIR, f"expC_{agent_name}_patterns.json")
    json.dump({k: {'mean': v[0], 'std': v[1], 'label': v[2]}
               for k, v in results.items()},
              open(save_path, 'w'), indent=2)
    print(f"Results saved to {save_path}")
    return results


def plot_experiment_C(results, agent_name="clean"):
    """Bar chart comparing failure patterns at fixed rate."""
    labels = [v[2] for v in results.values()]
    means  = [v[0] for v in results.values()]
    stds   = [v[1] for v in results.values()]

    colors = ['tab:red', 'tab:purple', 'tab:brown', 'tab:blue', 'tab:green']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), means, yerr=stds,
                  color=colors[:len(labels)], alpha=0.8,
                  capsize=5, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=12)
    ax.set_title(f'Failure Pattern Comparison at 20% Rate', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{mean:.0f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"expC_{agent_name}_patterns.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")



# ── Experiment D: Fine-grained failure rate sweep ─────────────────────────────

def experiment_D(agent, agent_name="clean"):
    """Fine-grained sweep: rates 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30"""
    print(f"\n{'='*60}")
    print(f"Experiment D: Fine-grained rate sweep | agent={agent_name}")
    print(f"{'='*60}")

    fine_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    modes = ['node_mask', 'edge_drop']
    results = {m: {} for m in modes}

    for mode in modes:
        for rate in fine_rates:
            mean_tt, std_tt = evaluate(agent, mode, rate)
            results[mode][rate] = (mean_tt, std_tt)
            print(f"  mode={mode:12s} rate={rate:.2f} | {mean_tt:.1f} +- {std_tt:.1f}s")

    import json
    json.dump({m: {str(r): v for r, v in rv.items()}
               for m, rv in results.items()},
              open(os.path.join(RESULTS_DIR, f"expD_{agent_name}_finegrain.json"), 'w'), indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'node_mask': 'tab:red', 'edge_drop': 'tab:blue'}
    labels = {'node_mask': 'Sensor Failure (node mask)', 'edge_drop': 'Comm. Failure (edge drop)'}
    for mode in modes:
        means = [results[mode][r][0] for r in fine_rates]
        stds  = [results[mode][r][1] for r in fine_rates]
        ax.plot(fine_rates, means, '-o', color=colors[mode], linewidth=2, label=labels[mode])
        ax.fill_between(fine_rates,
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.15, color=colors[mode])
    ax.set_xlabel('Failure Rate', fontsize=13)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=13)
    ax.set_title('Fine-Grained Failure Rate Sweep', fontsize=14)
    ax.set_xticks(fine_rates)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"expD_{agent_name}_finegrain.png"), dpi=150)
    plt.close()
    print(f"Exp D plot saved")
    return results


# ── Experiment E: Mixed failure ───────────────────────────────────────────────

def apply_mixed_failure(state, node_rate, edge_rate, rng):
    """Apply both node masking and edge dropping simultaneously."""
    from failure_modes import apply_node_masking, apply_edge_dropping
    features = state['features'].copy()
    adj      = state['adj'].copy()
    features, _ = apply_node_masking(features, node_rate, rng)
    adj,      _ = apply_edge_dropping(adj, edge_rate, rng)
    return {'features': features.astype(np.float32), 'adj': adj}


def evaluate_mixed(agent, node_rate, edge_rate, num_episodes=EVAL_EPISODES, seed=0):
    """Evaluate under simultaneous node masking + edge dropping."""
    
    travel_times = []
    device = next(agent.q_net.parameters()).device
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        env = make_env(num_steps=EVAL_STEPS)
        env.seed = seed + ep * 7
        state = env.reset()

        for step in range(EVAL_STEPS):
            obs = apply_mixed_failure(state, node_rate, edge_rate, rng)
            feat_t = torch.FloatTensor(obs['features']).unsqueeze(0).to(device)
            adj_t  = torch.LongTensor(obs['adj']).unsqueeze(0).to(device)
            with torch.no_grad():
                q, _ = agent.q_net(feat_t, adj_t)
            actions = q.squeeze(0).argmax(dim=-1).cpu().numpy()
            next_state, _, done, _ = env.step(actions)
            state = next_state
            if done:
                break

        travel_times.append(env.get_avg_travel_time())

    return np.mean(travel_times), np.std(travel_times)


def experiment_E(agent, agent_name="clean"):
    """Mixed failure: simultaneous node masking + edge dropping."""
    print(f"\n{'='*60}")
    print(f"Experiment E: Mixed failure | agent={agent_name}")
    print(f"{'='*60}")

    # Test combinations of node_rate and edge_rate
    combos = [
        (0.0, 0.0, 'Clean'),
        (0.2, 0.0, 'Sensor only (20%)'),
        (0.0, 0.2, 'Comm only (20%)'),
        (0.1, 0.1, 'Mixed (10%+10%)'),
        (0.2, 0.2, 'Mixed (20%+20%)'),
        (0.15, 0.15, 'Mixed (15%+15%)'),
    ]

    results = {}
    for node_r, edge_r, label in combos:
        mean_tt, std_tt = evaluate_mixed(agent, node_r, edge_r)
        results[label] = (mean_tt, std_tt)
        print(f"  {label:25s} | {mean_tt:.1f} +- {std_tt:.1f}s")

    import json
    json.dump({k: list(v) for k, v in results.items()},
              open(os.path.join(RESULTS_DIR, f"expE_{agent_name}_mixed.json"), 'w'), indent=2)

    # Plot bar chart
    labels = list(results.keys())
    means  = [results[l][0] for l in labels]
    stds   = [results[l][1] for l in labels]
    colors = ['gray', 'tab:red', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:orange']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), means, yerr=stds,
                  color=colors[:len(labels)], alpha=0.8,
                  capsize=5, edgecolor='black', linewidth=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.0f}s', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=12)
    ax.set_title('Mixed Failure: Simultaneous Sensor + Communication Failure', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"expE_{agent_name}_mixed.png"), dpi=150)
    plt.close()
    print(f"Exp E plot saved")
    return results


# ── Experiment F: Attention heads sweep ───────────────────────────────────────

def experiment_F():
    """Train nhead=1 vs nhead=4, compare under edge_drop failure."""
    print(f"\n{'='*60}")
    print(f"Experiment F: Attention heads sweep (nhead=1 vs 4)")
    print(f"{'='*60}")

    env = make_env()
    results = {}

    for nhead in [1, 4]:
        print(f"\n  Training nhead={nhead}...")
        agent = CoLightAgent(
            feature_dim=env.feature_dim,
            num_actions=max(env.num_phases),
            num_agents=env.num_intersections,
            top_k=env.TOP_K,
            lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
            epsilon_decay=0.95, batch_size=64, update_target_freq=5,
            embed_dim=64, att_dim=32, out_dim=64,
            nhead=nhead, num_layers=2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        injector = FailureInjector(mode='clean', failure_rate=0.0)
        for ep in range(50):  # shorter training for sweep
            state = env.reset()
            injector.reset(state)
            for step in range(STEPS_PER_EP):
                obs = injector.inject(state)
                actions = agent.select_actions(obs['features'], obs['adj'])
                next_state, rewards, done, _ = env.step(actions)
                next_obs = injector.inject(next_state)
                agent.store(obs, actions, rewards/20.0, next_obs, done)
                agent.train()
                state = next_state
                if done:
                    break
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.95)
            if (ep+1) % 10 == 0:
                print(f"    ep {ep+1}/50 | tt={env.get_avg_travel_time():.0f}s | eps={agent.epsilon:.2f}")

        # Evaluate under edge_drop at various rates
        nhead_results = {}
        for rate in [0.0, 0.1, 0.2, 0.3]:
            mean_tt, std_tt = evaluate(agent, 'edge_drop', rate)
            nhead_results[rate] = (mean_tt, std_tt)
            print(f"    edge_drop rate={rate:.1f} | {mean_tt:.1f} +- {std_tt:.1f}s")
        results[nhead] = nhead_results

    import json
    json.dump({str(k): {str(r): list(v) for r, v in rv.items()}
               for k, rv in results.items()},
              open(os.path.join(RESULTS_DIR, "expF_attn_heads.json"), 'w'), indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    rates = [0.0, 0.1, 0.2, 0.3]
    colors = {1: 'tab:orange', 4: 'tab:green'}
    for nhead in [1, 4]:
        means = [results[nhead][r][0] for r in rates]
        stds  = [results[nhead][r][1] for r in rates]
        ax.plot(rates, means, '-o', color=colors[nhead], linewidth=2,
                label=f'nhead={nhead}')
        ax.fill_between(rates,
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.15, color=colors[nhead])
    ax.set_xlabel('Edge Drop Rate', fontsize=13)
    ax.set_ylabel('Avg. Travel Time (s)', fontsize=13)
    ax.set_title('Attention Heads: nhead=1 vs nhead=4\nunder Communication Failure', fontsize=13)
    ax.set_xticks(rates)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "expF_attn_heads.png"), dpi=150)
    plt.close()
    print("Exp F plot saved")
    return results



# ── Experiment G: Mixed-failure robust training ───────────────────────────────

def train_mixed_robust(num_episodes=100, save_name='colight_mixed_robust'):
    """
    Train with simultaneous node_mask + edge_drop (10% each).
    Tests whether mixed-failure training yields better general robustness
    than single-failure training.
    """
    print(f"\n{'='*60}")
    print(f"Experiment G: Mixed-failure robust training")
    print(f"{'='*60}")

    from failure_modes import apply_node_masking, apply_edge_dropping
    env = make_env(num_steps=STEPS_PER_EP)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = CoLightAgent(
        feature_dim=env.feature_dim,
        num_actions=max(env.num_phases),
        num_agents=env.num_intersections,
        top_k=env.TOP_K,
        lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
        epsilon_decay=0.95, batch_size=64, update_target_freq=5,
        embed_dim=64, att_dim=32, out_dim=64, nhead=2, num_layers=2,
        device=device
    )

    rng = np.random.default_rng(42)
    episode_tts = []

    for ep in range(num_episodes):
        state = env.reset()
        for step in range(STEPS_PER_EP):
            # Mixed failure injection
            features = state['features'].copy()
            adj      = state['adj'].copy()
            features, _ = apply_node_masking(features, 0.1, rng)
            adj,      _ = apply_edge_dropping(adj, 0.1, rng)
            obs = {'features': features.astype(np.float32), 'adj': adj}

            actions    = agent.select_actions(obs['features'], obs['adj'])
            next_state, rewards, done, _ = env.step(actions)

            features2 = next_state['features'].copy()
            adj2      = next_state['adj'].copy()
            features2, _ = apply_node_masking(features2, 0.1, rng)
            adj2,      _ = apply_edge_dropping(adj2, 0.1, rng)
            next_obs = {'features': features2.astype(np.float32), 'adj': adj2}

            agent.store(obs, actions, rewards/20.0, next_obs, done)
            agent.train()
            state = next_state
            if done:
                break

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.95)
        tt = env.get_avg_travel_time()
        episode_tts.append(tt)
        if (ep+1) % 10 == 0:
            print(f"  ep {ep+1}/{num_episodes} | tt={tt:.1f}s | eps={agent.epsilon:.3f}")

    agent.save(os.path.join(MODEL_DIR, f"{save_name}.pt"))
    np.save(os.path.join(RESULTS_DIR, f"train_travel_times_{save_name}.npy"),
            np.array(episode_tts))
    print(f"Mixed-robust model saved")
    return agent


def experiment_G(mixed_agent, clean_agent):
    """
    Compare: clean-trained vs mixed-robust-trained
    under node_mask, edge_drop, and mixed failure.
    """
    print(f"\n{'='*60}")
    print(f"Experiment G: Mixed-robust vs Clean-trained comparison")
    print(f"{'='*60}")

    test_settings = [
        ('node_mask', 0.0), ('node_mask', 0.1), ('node_mask', 0.2), ('node_mask', 0.3),
        ('edge_drop', 0.0), ('edge_drop', 0.1), ('edge_drop', 0.2), ('edge_drop', 0.3),
    ]

    results = {'clean': {}, 'mixed_robust': {}}
    for agent, name in [(clean_agent, 'clean'), (mixed_agent, 'mixed_robust')]:
        for mode, rate in test_settings:
            mean_tt, std_tt = evaluate(agent, mode, rate)
            results[name][f"{mode}@{rate}"] = (mean_tt, std_tt)
            print(f"  {name:15s} {mode:10s} rate={rate:.1f} | {mean_tt:.1f} +- {std_tt:.1f}s")

    import json
    json.dump({n: {k: list(v) for k, v in rv.items()}
               for n, rv in results.items()},
              open(os.path.join(RESULTS_DIR, "expG_mixed_robust.json"), 'w'), indent=2)

    # Plot: 2 subplots, node_mask and edge_drop
    rates = [0.0, 0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode, title in zip(axes,
                                ['node_mask', 'edge_drop'],
                                ['Test: Sensor Failure', 'Test: Comm. Failure']):
        for name, color, label in [('clean', 'gray', 'Clean-trained'),
                                    ('mixed_robust', 'tab:purple', 'Mixed-robust-trained')]:
            means = [results[name][f"{mode}@{r}"][0] for r in rates]
            stds  = [results[name][f"{mode}@{r}"][1] for r in rates]
            ax.plot(rates, means, '-o', color=color, linewidth=2, label=label)
            ax.fill_between(rates,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            alpha=0.15, color=color)
        ax.set_xlabel('Failure Rate', fontsize=12)
        ax.set_ylabel('Avg. Travel Time (s)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(rates)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Mixed-Failure Robust Training vs Clean Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "expG_mixed_robust.png"), dpi=150)
    plt.close()
    print("Exp G plot saved")
    return results


# ── Experiment H: Second dataset (Hangzhou 4x4) ───────────────────────────────

import os as _os
_HANGZHOU_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "colight", "data", "Hangzhou", "4_4"
)

def make_hangzhou_env(num_steps=EVAL_STEPS):
    """Create environment with Hangzhou 4x4 road network."""
    from traffic_env import TSCEnv
    # Find roadnet and flow files
    files = os.listdir(_HANGZHOU_DIR)
    roadnet = [f for f in files if f.startswith('roadnet')][0]
    flow    = [f for f in files if f.endswith('.json') and 'roadnet' not in f][0]
    print(f"  Hangzhou: {roadnet}, {flow}")
    env = TSCEnv(
        data_dir=_HANGZHOU_DIR,
        roadnet_file=roadnet,
        flow_file=flow,
        num_steps=num_steps
    )
    return env


def experiment_H():
    """
    Train a fresh model on Hangzhou 4x4, then run failure sensitivity.
    Validates whether Jinan findings generalize to another road network.
    """
    print(f"\n{'='*60}")
    print(f"Experiment H: Hangzhou 4x4 generalization")
    print(f"{'='*60}")

    try:
        env_h = make_hangzhou_env(num_steps=STEPS_PER_EP)
    except Exception as e:
        print(f"  Could not load Hangzhou: {e}")
        return {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent_h = CoLightAgent(
        feature_dim=env_h.feature_dim,
        num_actions=max(env_h.num_phases),
        num_agents=env_h.num_intersections,
        top_k=env_h.TOP_K,
        lr=5e-4, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
        epsilon_decay=0.95, batch_size=64, update_target_freq=5,
        embed_dim=64, att_dim=32, out_dim=64, nhead=2, num_layers=2,
        device=device
    )

    injector = FailureInjector(mode='clean', failure_rate=0.0)
    print("  Training on Hangzhou (60 episodes)...")
    for ep in range(60):
        state = env_h.reset()
        injector.reset(state)
        for step in range(STEPS_PER_EP):
            obs     = injector.inject(state)
            actions = agent_h.select_actions(obs['features'], obs['adj'])
            next_state, rewards, done, _ = env_h.step(actions)
            next_obs = injector.inject(next_state)
            agent_h.store(obs, actions, rewards/20.0, next_obs, done)
            agent_h.train()
            state = next_state
            if done:
                break
        agent_h.epsilon = max(agent_h.epsilon_min, agent_h.epsilon * 0.95)
        if (ep+1) % 20 == 0:
            print(f"    ep {ep+1}/60 | tt={env_h.get_avg_travel_time():.1f}s | eps={agent_h.epsilon:.3f}")

    # Evaluate failure sensitivity on Hangzhou
    print("  Evaluating failure sensitivity on Hangzhou...")
    rates = [0.0, 0.1, 0.2, 0.3]
    results = {'node_mask': {}, 'edge_drop': {}}

    for mode in ['node_mask', 'edge_drop']:
        for rate in rates:
            # Custom eval for Hangzhou env
            tts = []
            for ep in range(2):
                env_eval = make_hangzhou_env(num_steps=EVAL_STEPS)
                env_eval.seed = ep * 7
                state = env_eval.reset()
                inj = FailureInjector(mode=mode, failure_rate=rate, seed=ep+1)
                inj.reset(state)
                for step in range(EVAL_STEPS):
                    obs = inj.inject(state)
                    feat_t = torch.FloatTensor(obs['features']).unsqueeze(0).to(device)
                    adj_t  = torch.LongTensor(obs['adj']).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q, _ = agent_h.q_net(feat_t, adj_t)
                    actions = q.squeeze(0).argmax(dim=-1).cpu().numpy()
                    state, _, done, _ = env_eval.step(actions)
                    if done:
                        break
                tts.append(env_eval.get_avg_travel_time())
            mean_tt, std_tt = float(np.mean(tts)), float(np.std(tts))
            results[mode][rate] = (mean_tt, std_tt)
            print(f"  Hangzhou mode={mode:10s} rate={rate:.1f} | {mean_tt:.1f} +- {std_tt:.1f}s")

    import json
    json.dump({m: {str(r): list(v) for r, v in rv.items()}
               for m, rv in results.items()},
              open(os.path.join(RESULTS_DIR, "expH_hangzhou.json"), 'w'), indent=2)

    # Plot comparison: Jinan vs Hangzhou
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'node_mask': 'tab:red', 'edge_drop': 'tab:blue'}
    labels = {'node_mask': 'Sensor Failure', 'edge_drop': 'Comm. Failure'}

    # Jinan data from Exp A
    jinan = {
        'node_mask': {0.0:599.8, 0.1:561.9, 0.2:559.0, 0.3:540.3},
        'edge_drop':  {0.0:599.8, 0.1:552.0, 0.2:534.1, 0.3:525.3},
    }

    for ax, (dataset, data_results), title in zip(
        axes,
        [('Jinan 3x4', jinan), ('Hangzhou 4x4', results)],
        ['Jinan 3×4', 'Hangzhou 4×4']
    ):
        for mode in ['node_mask', 'edge_drop']:
            if dataset == 'Jinan 3x4':
                means = [data_results[mode][r] for r in rates]
                stds  = [0]*4
            else:
                means = [data_results[mode][r][0] for r in rates]
                stds  = [data_results[mode][r][1] for r in rates]
            ax.plot(rates, means, '-o', color=colors[mode],
                    linewidth=2, label=labels[mode])
            ax.fill_between(rates,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            alpha=0.15, color=colors[mode])
        ax.set_xlabel('Failure Rate', fontsize=12)
        ax.set_ylabel('Avg. Travel Time (s)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(rates)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Generalization: Jinan vs Hangzhou Road Network', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "expH_hangzhou.png"), dpi=150)
    plt.close()
    print("Exp H plot saved")
    return results

def load_agent(env, model_name):
    agent = CoLightAgent(
        feature_dim=env.feature_dim,
        num_actions=max(env.num_phases),
        num_agents=env.num_intersections,
        top_k=env.TOP_K,
        **AGENT_KWARGS
    )
    path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    agent.load(path)
    print(f"Loaded model from {path}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train_clean', 'train_all', 'eval_all',
                                 'cross_failure', 'full', 'quick_test',
                                 'exp_c', 'exp_d', 'exp_e', 'exp_f',
                                 'exp_g', 'exp_h'])
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES)
    args = parser.parse_args()

    env = make_env()

    # ── Quick test: just make sure everything runs ──
    if args.mode == 'quick_test':
        print("Running quick test (2 episodes)...")
        agent, _ = train('clean', 0.0, num_episodes=2, save_name='colight_clean')
        results = experiment_A(agent, 'clean')
        plot_experiment_A(results, 'clean')
        print("\nQuick test passed!")
        return

    # ── Train clean ──
    if args.mode in ('train_clean', 'train_all', 'full'):
        agent_clean, _ = train(
            'clean', 0.0,
            num_episodes=args.episodes,
            save_name='colight_clean'
        )

    # ── Train with sensor failure ──
    if args.mode in ('train_all', 'full'):
        agent_node, _ = train(
            'node_mask', 0.2,
            num_episodes=args.episodes,
            save_name='colight_node_mask'
        )
        agent_edge, _ = train(
            'edge_drop', 0.2,
            num_episodes=args.episodes,
            save_name='colight_edge_drop'
        )

    # ── Load agents if not just trained ──
    if args.mode in ('eval_all', 'cross_failure'):
        agent_clean = load_agent(env, 'colight_clean')
        agent_node  = load_agent(env, 'colight_node_mask')
        agent_edge  = load_agent(env, 'colight_edge_drop')

    # ── Experiment A ──
    if args.mode in ('eval_all', 'full'):
        results_A = experiment_A(agent_clean, 'clean')
        plot_experiment_A(results_A, 'clean')

    # ── Experiment B ──
    if args.mode in ('cross_failure', 'full'):
        agents_dict = {
            'clean':     agent_clean,
            'node_mask': agent_node,
            'edge_drop': agent_edge,
        }
        results_B = experiment_B(agents_dict)
        plot_experiment_B(results_B)

    # ── Experiment C: failure pattern ──
    if args.mode in ('full', 'exp_c'):
        if args.mode == 'exp_c':
            agent_clean = load_agent(env, 'colight_clean')
        results_C = experiment_C(agent_clean, 'clean')
        plot_experiment_C(results_C, 'clean')

    # ── Training curves ──
    if args.mode == 'full':
        plot_training_curves(['colight_clean', 'colight_node_mask', 'colight_edge_drop'])

    # ── Experiment D: fine-grained sweep ──
    if args.mode in ('full', 'exp_d'):
        if args.mode == 'exp_d':
            agent_clean = load_agent(env, 'colight_clean')
        results_D = experiment_D(agent_clean, 'clean')

    # ── Experiment E: mixed failure ──
    if args.mode in ('full', 'exp_e'):
        if args.mode == 'exp_e':
            agent_clean = load_agent(env, 'colight_clean')
        results_E = experiment_E(agent_clean, 'clean')

    # ── Experiment F: attention heads ──
    if args.mode in ('full', 'exp_f'):
        results_F = experiment_F()

    # ── Experiment G: mixed robust training ──
    if args.mode in ('exp_g',):
        agent_clean = load_agent(env, 'colight_clean')
        mixed_agent = train_mixed_robust(num_episodes=100)
        results_G = experiment_G(mixed_agent, agent_clean)

    # ── Experiment H: Hangzhou generalization ──
    if args.mode in ('exp_h',):
        results_H = experiment_H()

    print("\nAll done! Results in ./results/")


if __name__ == '__main__':
    main()