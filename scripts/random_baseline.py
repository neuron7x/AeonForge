#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tanh")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log-jsonl", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    S, A = args.state_dim, args.action_dim
    # Simple synthetic learning curve for a random policy (monotonic noise as a sanity baseline)
    rewards = np.cumsum(rng.normal(0.0, 0.1, size=args.iters)) - np.linspace(0, 1.0, args.iters)
    rmses = np.abs(rng.normal(0.5, 0.15, size=args.iters))

    if args.log_jsonl:
        p = Path(args.log_jsonl)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for i, (r, rmse) in enumerate(zip(rewards, rmses), start=1):
                rec = {"it": i, "reward": float(r), "rmse": float(max(0.0, rmse))}
                f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
