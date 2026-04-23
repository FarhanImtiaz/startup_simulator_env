# Multi-Agent Startup Simulator (MASS)

This project implements a startup simulation environment inspired by `Project_Overview.md`.
It now includes:

- a startup world with multi-agent decision flow
- hidden market variables
- partial observability and noisy signals
- stochastic marketing outcomes
- random external events
- delayed action consequences
- global and agent-specific rewards
- multi-episode evaluation and saved metrics
- OpenEnv-style wrapper utilities
- prompt scaffolding for future LLM agents

## Files

- `environment.py` - startup world, hidden state, events, delayed effects, reward logic.
- `agents.py` - heuristic CEO, Tech, Growth, and Finance roles that act on noisy observations.
- `simulate.py` - runs episodes, supports CLI flags, and can save an episode summary.
- `evaluation.py` - runs many episodes, aggregates baseline metrics, and saves JSON/CSV outputs.
- `openenv_wrapper.py` - training-friendly reset/step wrapper around the environment.
- `llm_agents.py` - prompt templates, action parsing, and fallback-safe LLM agent scaffolding.
- `train.py` - collects trajectories in a training-ready format without running optimization.
- `Project_Overview.md` - the product and hackathon blueprint the code is following.

## Run

```bash
python3 simulate.py
```

Evaluate multiple episodes and save metrics:

```bash
python3 evaluation.py --episodes 10 --horizon 30 --save-dir outputs
```

Collect trajectories for future training:

```bash
python3 train.py --episodes 20 --horizon 30 --output outputs/trajectories.json
```

## What Is Next

- clean `.gitignore` and repo clutter
- add plots / visual summaries for reward and survival
- connect `llm_agents.py` to a real model backend
- run actual training and before-vs-after comparison when compute is available
