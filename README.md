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

Save a single episode summary:

```bash
python3 simulate.py --quiet --save-summary outputs/single_episode.json
```

Collect trajectories for later training work:

```bash
python3 train.py --episodes 20 --horizon 30 --output outputs/trajectories.json
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
- `outputs/trajectories_prompt.json`

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
- prompt scaffolding for future LLM control

What is not implemented yet:

- real model backend integration in `llm_agents.py`
- policy optimization or fine-tuning
- before-vs-after training comparison
- polished visualization and demo packaging

## Known Limitation

`evaluation.py` is close, but the current step-level CSV export path is not fully aligned with the richer per-step log format produced by `simulate.py`.

In practice:

- `simulate.py` works
- `train.py` works for trajectory collection
- `evaluation.py` may need a small export fix before relying on CSV step dumps

## If You Want To Understand The Codebase

Start with:

1. `README.md`
2. `simulate.py`
3. `agents.py`
4. `environment.py`

Then use `TEMP_CODEBASE_GUIDE.md` for the detailed file-by-file and function-by-function walkthrough.

## Next Good Improvements

- fix the `evaluation.py` step CSV export mismatch
- add tests for environment transitions and reward behavior
- make action costs and reward weights configurable
- improve disagreement quality between heuristic agents
- connect `llm_agents.py` to a real model backend
- add charts or visual summaries for demo use
