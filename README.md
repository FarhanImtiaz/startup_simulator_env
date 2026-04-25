# Multi-Agent Startup Simulator (MASS)

`MASS` is a compact startup simulation environment where specialized agents make business decisions under uncertainty over multiple timesteps.

The current project includes:

- a shared startup environment with hidden market state
- partial observability and noisy signals
- delayed action consequences
- stochastic marketing outcomes and random external events
- heuristic Tech, Growth, Finance, and CEO agents
- prompt scaffolding for future LLM-driven agents
- single-episode simulation, multi-episode evaluation, and trajectory collection scripts
- a lightweight training-style wrapper around the environment

## Project Goal

The repo is building toward a world-modeling environment for:

- multi-agent coordination
- long-horizon decision-making
- business-style tradeoffs between growth, product quality, and financial survival

Right now, the codebase is best understood as:

- a working simulation prototype
- a heuristic baseline
- an LLM integration scaffold
- a trajectory collection foundation for future training

## How The Simulation Works

At each step:

1. The environment exposes a noisy observation of the startup state.
2. Three co-founders propose actions:
   - Tech Co-founder
   - Growth Co-founder
   - Finance Co-founder
3. The CEO chooses one final action.
4. The environment applies:
   - delayed effects from previous actions
   - the chosen action
   - recurring startup dynamics
   - a possible random external event
5. The environment computes reward and returns the next observation.

The startup can succeed or fail based on product quality, user growth, cash runway, burn, market conditions, and random shocks.

## Main Files

- `environment.py`: core startup world, hidden state, delayed effects, events, rewards, termination.
- `agents.py`: heuristic co-founder policies and CEO decision logic.
- `simulate.py`: single-episode runner with CLI flags and optional saved summary.
- `evaluation.py`: multi-episode runner with aggregate metrics and export helpers.
- `llm_agents.py`: prompt builders, safe action parsing, and fallback-to-heuristic LLM scaffolding.
- `openenv_wrapper.py`: minimal reset/step wrapper for training-style integrations.
- `train.py`: trajectory collection script for future optimization workflows.
- `Project_Overview.md`: original hackathon/product framing.
- `TEMP_IMPLEMENTATION_CHECKLIST.md`: temporary progress tracker.
- `TEMP_CODEBASE_GUIDE.md`: temporary deep codebase walkthrough for development and onboarding.

## Quick Start

Run a single episode:

```bash
python3 simulate.py
```

Run a short verbose simulation with hidden state debugging:

```bash
python3 simulate.py --horizon 10 --show-hidden-state
```

Use the full old-style trace when you want every reward component and full reasoning:

```bash
python3 simulate.py --horizon 10 --log-detail full
```

Save a single episode summary:

```bash
python3 simulate.py --quiet --save-summary outputs/single_episode.json
```

Run a baseline evaluation with report artifacts:

```bash
python3 evaluation.py --episodes 20 --horizon 30 --save-dir outputs/eval
```

Collect trajectories for later training work:

```bash
python3 train.py --episodes 20 --horizon 30 --output outputs/trajectories.json
```

Export training-ready CEO decision datasets:

```bash
python3 train.py \
  --episodes 20 \
  --horizon 30 \
  --output outputs/trajectories.json \
  --sft-output outputs/ceo_sft.jsonl \
  --preference-output outputs/ceo_preferences.jsonl
```

## Agent Modes

Two agent modes are currently supported in the simulation scripts:

- `heuristic`: uses the hand-written policies in `agents.py`
- `prompt_scaffold`: uses prompt-based wrappers from `llm_agents.py`, but still falls back safely because no real model backend is wired in by default

Example:

```bash
python3 simulate.py --agent-mode prompt_scaffold
```

## Outputs

Typical generated outputs include:

- `outputs/single_episode.json`
- `outputs/trajectories.json`
- `outputs/ceo_sft.jsonl`
- `outputs/ceo_preferences.jsonl`
- `outputs/trajectories_prompt.json`
- `outputs/eval/evaluation_summary.json`
- `outputs/eval/episode_metrics.csv`
- `outputs/eval/step_metrics.csv`
- `outputs/eval/action_distribution.csv`
- `outputs/eval/baseline_report.md`
- `outputs/eval/reward_curve.svg`
- `outputs/eval/outcome_curve.svg`
- `outputs/eval/action_distribution.svg`

These are useful for inspecting step-by-step behavior and bootstrapping future training or evaluation work.

## Current State

What is implemented:

- startup environment with public and hidden state
- delayed effects queue
- random event system
- shaped company reward
- role-based heuristic decision flow
- single-episode simulation
- trajectory collection
- SFT and preference dataset export for CEO decision training
- prompt scaffolding for future LLM control

What is not implemented yet:

- real model backend integration in `llm_agents.py`
- policy optimization or fine-tuning
- before-vs-after training comparison
- polished visualization and demo packaging

## Known Limitation

The Phase 2 evaluation flow now exports flat CSVs, a Markdown baseline report, and lightweight SVG plots without requiring plotting dependencies.

## If You Want To Understand The Codebase

Start with:

1. `README.md`
2. `simulate.py`
3. `agents.py`
4. `environment.py`

Then use `TEMP_CODEBASE_GUIDE.md` for the detailed file-by-file and function-by-function walkthrough.

## Next Good Improvements

- add tests for environment transitions and reward behavior
- make action costs and reward weights configurable
- improve disagreement quality between heuristic agents
- connect `llm_agents.py` to a real model backend
- add charts or visual summaries for demo use
