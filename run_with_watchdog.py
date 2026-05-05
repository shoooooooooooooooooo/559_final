"""
Auto-restart wrapper for train_eval.py
Detects when output is stuck and auto-restarts from where it left off.
Usage: python run_with_watchdog.py --mode cross_failure
"""

import subprocess
import time
import sys
import os
import re
import argparse

TIMEOUT = 300  # seconds with no output before restart
MODE = 'cross_failure'

def run_with_watchdog(mode, timeout=TIMEOUT):
    print(f"\n{'='*60}")
    print(f"Watchdog started | mode={mode} | timeout={timeout}s")
    print(f"Auto-restarts if stuck for {timeout}s")
    print(f"{'='*60}\n")

    results = {}  # store completed results
    attempt = 0

    while True:
        attempt += 1
        print(f"\n[Watchdog] Attempt {attempt} | Collected {len(results)} results so far")

        cmd = ['python', '-u', 'train_eval.py', '--mode', mode]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        last_output_time = time.time()
        lines_this_run = 0

        try:
            while True:
                # Check if process finished
                if proc.poll() is not None:
                    # Read remaining output
                    remaining = proc.stdout.read()
                    for line in remaining.splitlines():
                        _process_line(line, results)
                    print(f"\n[Watchdog] Process finished normally!")
                    _print_summary(results)
                    return results

                # Try to read a line with timeout
                line = proc.stdout.readline()
                if line:
                    last_output_time = time.time()
                    lines_this_run += 1
                    line = line.rstrip()
                    _process_line(line, results)

                    # Print progress
                    if any(k in line for k in ['mode=', 'train=', 'Episode', 'saved', 'done', 'Failure', 'nhead', '+-', 'rate=']):
                        print(line)
                    elif 'Controllable' in line:
                        print('.', end='', flush=True)

                # Check timeout
                elapsed = time.time() - last_output_time
                if elapsed > timeout:
                    print(f"\n[Watchdog] No output for {elapsed:.0f}s — restarting...")
                    proc.kill()
                    proc.wait()
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n[Watchdog] Interrupted by user")
            proc.kill()
            _print_summary(results)
            return results

    return results


def _process_line(line, results):
    """Extract travel time results from output lines."""
    # Match: mode=X  rate=Y | travel_time=Z ± W
    m = re.search(
        r'mode=(\S+)\s+rate=([\d.]+)\s+\|\s+travel_time=([\d.]+)\s+±\s+([\d.]+)',
        line
    )
    if m:
        mode, rate, tt, std = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        results[f"{mode}@{rate}"] = (tt, std)

    # Match cross-failure: train=X  test=Y  rate=Z | ...
    m2 = re.search(
        r'train=(\S+)\s+test=(\S+)\s+rate=([\d.]+)\s+\|\s+([\d.]+)\s+±\s+([\d.]+)',
        line
    )
    if m2:
        train, test, rate = m2.group(1), m2.group(2), m2.group(3)
        tt, std = float(m2.group(4)), float(m2.group(5))
        results[f"{train}->{test}@{rate}"] = (tt, std)


def _print_summary(results):
    if not results:
        print("\n[Watchdog] No results collected.")
        return
    print(f"\n{'='*60}")
    print(f"COLLECTED RESULTS ({len(results)} entries):")
    print(f"{'='*60}")
    for key, (tt, std) in sorted(results.items()):
        print(f"  {key:40s} | {tt:.1f} ± {std:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='cross_failure',
                        choices=['eval_all', 'cross_failure', 'exp_c', 'exp_d', 'exp_e', 'exp_f', 'exp_g', 'exp_h', 'all_new', 'all_gh'])
    parser.add_argument('--timeout', type=int, default=300,
                        help='Seconds without output before restart')
    args = parser.parse_args()

    if args.mode == 'all_new':
        all_results = {}
        for m in ['exp_d', 'exp_e', 'exp_f']:
            print(f"\n>>> Running {m}...")
            r = run_with_watchdog(m, args.timeout)
            all_results.update(r)
        print("\n=== ALL EXPERIMENTS DONE ===")
        for k, v in sorted(all_results.items()):
            print(f"  {k:45s} | {v[0]:.1f} +- {v[1]:.1f}s")
    elif args.mode == 'all_gh':
        for m in ['exp_g', 'exp_h']:
            print(f"\n>>> Running {m}...")
            run_with_watchdog(m, args.timeout)
        print("\n=== G and H DONE ===")
    else:
        results = run_with_watchdog(args.mode, args.timeout)