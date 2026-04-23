from environment import StartupEnvironment
from agents import TechCoFounder, GrowthCoFounder, FinanceCoFounder, CEO


def run_episode(env: StartupEnvironment, horizon: int = 30, verbose: bool = True) -> float:
    env.reset()
    tech = TechCoFounder()
    growth = GrowthCoFounder()
    finance = FinanceCoFounder()
    ceo = CEO()

    total_reward = 0.0
    for _ in range(horizon):
        observation = env.get_observation()
        proposals = {
            tech.name: tech.propose(observation),
            growth.name: growth.propose(observation),
            finance.name: finance.propose(observation),
        }

        selected = ceo.choose_action(proposals, observation)
        result = env.step(selected.action)
        total_reward += result["reward"]

        if verbose:
            print(f"Day {observation['day']}: CEO chose {selected.action}")
            print(f"  proposals: {[f'{name} -> {proposal.action}' for name, proposal in proposals.items()]}")
            print(f"  reward={result['reward']}, money={result['state']['money']}, users={result['state']['users']}, quality={result['state']['product_quality']}")
            print("---")

        if result["done"]:
            break

    if verbose:
        print(f"Episode finished day={env.state.day}, total_reward={round(total_reward,3)}")
    return round(total_reward, 3)


def main() -> None:
    env = StartupEnvironment(max_days=30)
    run_episode(env, horizon=30, verbose=True)


if __name__ == "__main__":
    main()
