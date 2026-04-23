import argparse
import json
from pathlib import Path
from typing import Dict, List

from environment import StartupEnvironment
from simulate import run_episode


def collect_trajectories(
    episodes: int = 20,
    horizon: int = 30,
    base_seed: int = 7,
    agent_mode: str = "heuristic",
) -> List[Dict[str, object]]:
    trajectories: List[Dict[str, object]] = []
    for episode_index in range(episodes):
        env = StartupEnvironment(max_days=horizon, seed=base_seed + episode_index)
        summary = run_episode(
            env,
            horizon=horizon,
            verbose=False,
            show_hidden_state=False,
            agent_mode=agent_mode,
        )

        trajectories.append(
            {
                "episode_index": episode_index,
                "agent_mode": agent_mode,
                "total_reward": summary["total_reward"],
                "days_completed": summary["days_completed"],
                "final_state": summary["final_state"],
                "steps": summary["episode_log"],
            }
        )

    return trajectories


def save_trajectories(path: str, trajectories: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect startup simulator trajectories for future training runs."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold"], default="heuristic")
    parser.add_argument("--output", default="outputs/trajectories.json")
    args = parser.parse_args()

    trajectories = collect_trajectories(
        episodes=args.episodes,
        horizon=args.horizon,
        base_seed=args.seed,
        agent_mode=args.agent_mode,
    )
    save_trajectories(args.output, trajectories)

    print(
        "Collected trajectories"
        f" episodes={len(trajectories)}"
        f" output={args.output}"
    )
    print("Training optimization is intentionally left out here because it requires model compute resources.")


if __name__ == "__main__":
    main()
