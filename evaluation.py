import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from simulate import run_episode
from environment import StartupEnvironment


def evaluate(
    episodes: int = 10,
    horizon: int = 30,
    base_seed: int = 7,
    agent_mode: str = "heuristic",
    save_dir: Optional[str] = "outputs",
) -> Dict[str, object]:
    episode_summaries: List[Dict[str, object]] = []

    for episode_index in range(episodes):
        env = StartupEnvironment(max_days=horizon, seed=base_seed + episode_index)
        summary = run_episode(
            env,
            horizon=horizon,
            verbose=False,
            show_hidden_state=False,
            agent_mode=agent_mode,
        )
        summary["episode_index"] = episode_index
        episode_summaries.append(summary)

    aggregate = _build_aggregate_metrics(episode_summaries)
    payload = {
        "config": {
            "episodes": episodes,
            "horizon": horizon,
            "base_seed": base_seed,
            "agent_mode": agent_mode,
        },
        "aggregate": aggregate,
        "episodes": episode_summaries,
    }

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_json(output_dir / "evaluation_summary.json", payload)
        _save_episode_csv(output_dir / "episode_metrics.csv", episode_summaries)
        _save_step_csv(output_dir / "step_metrics.csv", episode_summaries)

    return payload


def _build_aggregate_metrics(episode_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    total_rewards = [summary["total_reward"] for summary in episode_summaries]
    final_money = [summary["final_state"]["money"] for summary in episode_summaries]
    final_users = [summary["final_state"]["users"] for summary in episode_summaries]
    survival_flags = [summary["final_state"]["money"] >= 0 and summary["final_state"]["users"] > 0 for summary in episode_summaries]
    decision_efficiency = [
        _positive_reward_ratio(summary["episode_log"])
        for summary in episode_summaries
    ]
    growth_consistency = [
        _growth_consistency(summary["episode_log"])
        for summary in episode_summaries
    ]

    return {
        "episodes": len(episode_summaries),
        "average_total_reward": round(mean(total_rewards), 3),
        "average_final_money": round(mean(final_money), 3),
        "average_final_users": round(mean(final_users), 3),
        "survival_rate": round(sum(1 for flag in survival_flags if flag) / max(1, len(survival_flags)), 3),
        "growth_consistency": round(mean(growth_consistency), 3),
        "decision_efficiency": round(mean(decision_efficiency), 3),
        "best_episode_reward": round(max(total_rewards), 3),
        "worst_episode_reward": round(min(total_rewards), 3),
    }


def _positive_reward_ratio(episode_log: List[Dict[str, object]]) -> float:
    if not episode_log:
        return 0.0
    positive_steps = sum(1 for row in episode_log if row["reward"] > 0)
    return positive_steps / len(episode_log)


def _growth_consistency(episode_log: List[Dict[str, object]]) -> float:
    if len(episode_log) < 2:
        return 0.0
    non_crash_steps = 0
    for previous, current in zip(episode_log, episode_log[1:]):
        if current["users"] >= previous["users"] * 0.85:
            non_crash_steps += 1
    return non_crash_steps / (len(episode_log) - 1)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_episode_csv(path: Path, episode_summaries: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "days_completed",
                "total_reward",
                "final_money",
                "final_users",
                "final_quality",
            ],
        )
        writer.writeheader()
        for summary in episode_summaries:
            writer.writerow(
                {
                    "episode_index": summary["episode_index"],
                    "days_completed": summary["days_completed"],
                    "total_reward": summary["total_reward"],
                    "final_money": summary["final_state"]["money"],
                    "final_users": summary["final_state"]["users"],
                    "final_quality": summary["final_state"]["product_quality"],
                }
            )


def _save_step_csv(path: Path, episode_summaries: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "day",
                "chosen_action",
                "reward",
                "event",
                "money",
                "users",
                "quality",
            ],
        )
        writer.writeheader()
        for summary in episode_summaries:
            for row in summary["episode_log"]:
                writer.writerow({"episode_index": summary["episode_index"], **row})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the startup simulator across many episodes.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold"], default="heuristic")
    parser.add_argument("--save-dir", default="outputs")
    args = parser.parse_args()

    payload = evaluate(
        episodes=args.episodes,
        horizon=args.horizon,
        base_seed=args.seed,
        agent_mode=args.agent_mode,
        save_dir=args.save_dir,
    )

    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
