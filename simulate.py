import argparse
import json
from pathlib import Path

from agents import build_heuristic_agents
from environment import StartupEnvironment
from llm_agents import build_prompted_agents


def run_episode(
    env: StartupEnvironment,
    horizon: int = 30,
    verbose: bool = True,
    show_hidden_state: bool = False,
    agent_mode: str = "heuristic",
) -> dict:
    observation = env.reset()
    tech, growth, finance, ceo = _build_agent_stack(agent_mode)

    total_reward = 0.0
    episode_log = []

    for _ in range(horizon):
        proposals = {
            tech.name: tech.propose(observation),
            growth.name: growth.propose(observation),
            finance.name: finance.propose(observation),
        }

        selected = ceo.choose_action(proposals, observation)
        result = env.step(selected.action, proposals=proposals)
        total_reward += result["reward"]

        log_row = {
            "day": observation["day"],
            "observation": dict(observation),
            "proposals": {
                name: {
                    "action": proposal.action,
                    "reasoning": proposal.reasoning,
                }
                for name, proposal in proposals.items()
            },
            "chosen_action": selected.action,
            "chosen_reasoning": selected.reasoning,
            "reward": result["reward"],
            "event": result["event"],
            "money": result["state"]["money"],
            "users": result["state"]["users"],
            "quality": result["state"]["product_quality"],
        }
        prompt_debug = _collect_prompt_debug(tech, growth, finance, ceo)
        if prompt_debug:
            log_row["prompt_debug"] = prompt_debug
        episode_log.append(log_row)

        if verbose:
            print(f"Day {observation['day']}: CEO chose {selected.action}")
            for name, proposal in proposals.items():
                print(f"  {name}: {proposal.action} | {proposal.reasoning}")
            print(
                "  obs:"
                f" growth={observation['recent_user_growth']},"
                f" ad={observation['ad_performance']},"
                f" runway_hint={observation['runway_hint']},"
                f" events={observation['recent_events']}"
            )
            print(
                "  result:"
                f" reward={result['reward']},"
                f" event={result['event']},"
                f" money={result['state']['money']},"
                f" users={result['state']['users']},"
                f" quality={result['state']['product_quality']}"
            )
            print(f"  agent_rewards={result['agent_rewards']}")
            if show_hidden_state:
                print(f"  debug_hidden={result['debug_state']['hidden_state']}")
                print(f"  pending_effects={result['debug_state']['pending_effects']}")
            print("---")

        observation = result["state"]
        if result["done"]:
            break

    summary = {
        "total_reward": round(total_reward, 3),
        "days_completed": len(episode_log),
        "final_state": env.get_debug_state()["public_state"],
        "episode_log": episode_log,
    }

    if verbose:
        print(
            "Episode finished"
            f" day={summary['days_completed']},"
            f" total_reward={summary['total_reward']},"
            f" final_money={summary['final_state']['money']},"
            f" final_users={summary['final_state']['users']}"
        )

    return summary


def _build_agent_stack(agent_mode: str):
    if agent_mode == "prompt_scaffold":
        return build_prompted_agents()
    return build_heuristic_agents()


def _collect_prompt_debug(tech, growth, finance, ceo):
    prompt_debug = {}
    for agent in (tech, growth, finance, ceo):
        prompt = getattr(agent, "last_prompt", None)
        raw_response = getattr(agent, "last_raw_response", None)
        if prompt is None and raw_response is None:
            continue
        prompt_debug[agent.name] = {
            "system_prompt": None if prompt is None else prompt.system_prompt,
            "user_prompt": None if prompt is None else prompt.user_prompt,
            "raw_response": raw_response,
        }
    return prompt_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Run startup simulator episodes.")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show-hidden-state", action="store_true")
    parser.add_argument("--agent-mode", choices=["heuristic", "prompt_scaffold"], default="heuristic")
    parser.add_argument("--save-summary", type=str, default=None)
    args = parser.parse_args()

    env = StartupEnvironment(max_days=args.horizon, seed=args.seed)
    summary = run_episode(
        env,
        horizon=args.horizon,
        verbose=not args.quiet,
        show_hidden_state=args.show_hidden_state,
        agent_mode=args.agent_mode,
    )

    if args.save_summary:
        path = Path(args.save_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
