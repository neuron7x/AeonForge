#!/usr/bin/env python3
import argparse, subprocess, sys, itertools, json, os
from pathlib import Path

def run_one(env, seed, model, meta, linucb_mode, iters, horizon, state_dim, action_dim, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"log_env={env}_seed={seed}_model={model}_meta={meta}_mode={linucb_mode}.jsonl"
    cmd = [
        sys.executable, "-m", "agi_core.engine.runner",
        "--env", env,
        "--iters", str(iters),
        "--horizon", str(horizon),
        "--state-dim", str(state_dim),
        "--action-dim", str(action_dim),
        "--seed", str(seed),
        "--model", model,
        "--meta", meta,
        "--linucb-mode", linucb_mode,
        "--log-jsonl", str(log_path),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)
    return log_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="tanh,lqr,reacher,minigrid", help="comma list")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="comma list")
    ap.add_argument("--models", default="linear,neural", help="comma list")
    ap.add_argument("--metas", default="linucb,bandit", help="comma list")
    ap.add_argument("--linucb-modes", default="diag,full", help="comma list")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--out", default="out/ablations")
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    metas = [m.strip() for m in args.metas.split(",") if m.strip()]
    modes = [m.strip() for m in args.linucb_modes.split(",") if m.strip()]

    logs = []
    for env, seed, model, meta, mode in itertools.product(envs, seeds, models, metas, modes):
        try:
            lp = run_one(env, seed, model, meta, mode, args.iters, args.horizon, args.state_dim, args.action_dim, args.out)
            logs.append(str(lp))
            print("OK:", lp)
        except SystemExit:
            # keep going to finish the sweep
            pass

    # Emit an index file so downstream tools can discover logs easily
    index_path = Path(args.out) / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"logs": logs}, f, indent=2)
    print("Wrote", index_path)

if __name__ == "__main__":
    main()
