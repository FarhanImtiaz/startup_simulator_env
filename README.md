# Multi-Agent Startup Simulator (MASS)

This project implements a startup simulation environment inspired by the Project Overview.
The environment includes a CEO decision-maker and specialized co-founder agents.

## Files

- `environment.py` — core startup state, transition logic, reward model.
- `agents.py` — simple Tech, Growth, Finance co-founder strategies and CEO decision logic.
- `simulate.py` — runs a single episode with proposals and CEO selection.

## Run

From the project directory:

```powershell
python simulate.py
```

## Next steps

- add an OpenEnv-compatible wrapper
- replace heuristic agents with LLM prompts
- implement training with TRL or Unsloth
