#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def load_series(log_path: Path):
    xs, rs, es = [], [], []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            xs.append(int(rec.get("it", len(xs)+1)))
            rs.append(float(rec.get("reward", 0.0)))
            es.append(float(rec.get("rmse", 0.0)))
    return xs, rs, es

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="comma separated JSONL files")
    ap.add_argument("--out", default="out/plots")
    ap.add_argument("--title", default="Training Curves")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for log in [Path(x.strip()) for x in args.logs.split(",") if x.strip()]:
        xs, rs, es = load_series(log)
        plt.figure()
        plt.plot(xs, rs)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title(args.title + f" — {log.stem}")
        fig1 = out / (log.stem + "_reward.png")
        plt.savefig(fig1, bbox_inches="tight", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(xs, es)
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.title(args.title + f" — {log.stem}")
        fig2 = out / (log.stem + "_rmse.png")
        plt.savefig(fig2, bbox_inches="tight", dpi=160)
        plt.close()

        print("Wrote", fig1, "and", fig2)

if __name__ == "__main__":
    main()
