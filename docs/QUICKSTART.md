# Quickstart

```bash
# 1) Install (dev extras include tests & linters)
pip install -e .[dev,viz]

# 2) Smoke run + plot
python -m agi_core.engine.runner --env tanh --iters 10 --horizon 10 --log-jsonl out/quickstart.jsonl
python scripts/plot_results.py --logs out/quickstart.jsonl --out out/plots --title "Quickstart"

# 3) Compare to random baseline
python scripts/compare_baselines.py --env tanh --iters 20 --horizon 20 --out out/baselines

# 4) Full ablation sweep
python scripts/run_ablations.py --envs tanh,lqr --seeds 0,1,2 --models linear,neural --metas linucb,bandit --linucb-modes diag --iters 50 --horizon 30 --out out/ablations
```
