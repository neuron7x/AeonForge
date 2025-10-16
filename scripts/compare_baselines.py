#!/usr/bin/env python3
import argparse, subprocess, sys, json, csv
from pathlib import Path

def parse_jsonl(path: Path):
    xs, rs = [], []
    if not path.exists():
        return xs, rs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                xs.append(int(rec.get("it", len(xs)+1)))
                rs.append(float(rec.get("reward", 0.0)))
            except Exception:
                pass
    return xs, rs

def summarize(rs):
    if not rs: return {"final": 0.0, "best": 0.0, "avg": 0.0}
    import numpy as np
    arr = np.array(rs, dtype=float)
    return {"final": float(arr[-1]), "best": float(arr.max()), "avg": float(arr.mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tanh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--model", default="linear", choices=["linear","neural"])
    ap.add_argument("--meta", default="linucb", choices=["linucb","bandit"])
    ap.add_argument("--linucb-mode", default="diag", choices=["diag","full"])
    ap.add_argument("--out", default="out/baselines")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_agent = out / "agent.jsonl"
    log_rand  = out / "random.jsonl"

    # Agent via existing CLI runner
    cmd = [
        sys.executable, "-m", "agi_core.engine.runner",
        "--env", args.env, "--iters", str(args.iters), "--horizon", str(args.horizon),
        "--state-dim", str(args.state_dim), "--action-dim", str(args.action_dim),
        "--seed", str(args.seed), "--model", args.model, "--meta", args.meta,
        "--linucb-mode", args.linucb_mode, "--log-jsonl", str(log_agent)
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)

    # Random baseline
    p2 = subprocess.run([sys.executable, "scripts/random_baseline.py",
        "--env", args.env, "--iters", str(args.iters), "--horizon", str(args.horizon),
        "--state-dim", str(args.state_dim), "--action-dim", str(args.action_dim),
        "--seed", str(args.seed), "--log-jsonl", str(log_rand)], text=True, capture_output=True)
    if p2.returncode != 0:
        print(p2.stdout); print(p2.stderr, file=sys.stderr)
        raise SystemExit(p2.returncode)

    # Summaries
    _, r_agent = parse_jsonl(log_agent)
    _, r_rand  = parse_jsonl(log_rand)
    s_agent, s_rand = summarize(r_agent), summarize(r_rand)

    # Save CSV
    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","agent","random"])
        for k in ["final","best","avg"]:
            w.writerow([k, s_agent[k], s_rand[k]])
    print("Wrote", csv_path)

if __name__ == "__main__":
    main()
