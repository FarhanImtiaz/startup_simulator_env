import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StartupState:
    day: int = 1
    money: float = 100000.0
    users: int = 100
    product_quality: float = 0.5
    team_size: int = 2
    burn_rate: float = 12000.0
    market_demand: float = 0.7
    recent_actions: List[str] = field(default_factory=list)
    last_users: int = 100


class StartupEnvironment:
    ACTIONS = [
        "hire_employee",
        "fire_employee",
        "invest_in_product",
        "run_marketing_campaign",
        "do_nothing",
        "pivot_strategy",
    ]

    def __init__(self, max_days: int = 30):
        self.max_days = max_days
        self.state = StartupState()
        self.reset()

    def reset(self) -> StartupState:
        self.state = StartupState()
        self.state.recent_actions = []
        self.state.last_users = self.state.users
        return self.state

    def get_observation(self) -> Dict[str, object]:
        return {
            "day": self.state.day,
            "money": round(self.state.money, 2),
            "users": self.state.users,
            "product_quality": round(self.state.product_quality, 3),
            "team_size": self.state.team_size,
            "burn_rate": round(self.state.burn_rate, 2),
            "market_demand": round(self.state.market_demand, 3),
            "recent_actions": list(self.state.recent_actions[-5:]),
        }

    def step(self, action: str) -> Dict[str, object]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.state.last_users = self.state.users
        self._apply_action(action)
        self._apply_environment_dynamics()

        reward = self._compute_reward()
        done = self._is_done()
        self.state.day += 1
        self.state.recent_actions.append(action)

        return {
            "state": self.get_observation(),
            "reward": reward,
            "done": done,
            "action": action,
        }

    def _apply_action(self, action: str) -> None:
        if action == "hire_employee":
            cost = 12000
            self.state.money -= cost
            self.state.team_size += 1
            self.state.burn_rate += 2500

        elif action == "fire_employee":
            if self.state.team_size > 1:
                self.state.team_size -= 1
                self.state.money += 5000
                self.state.burn_rate = max(6000, self.state.burn_rate - 2000)

        elif action == "invest_in_product":
            cost = 9000
            self.state.money -= cost
            self.state.product_quality += 0.08 + 0.01 * self.state.team_size
            self.state.product_quality = min(1.0, self.state.product_quality)

        elif action == "run_marketing_campaign":
            cost = 7000
            self.state.money -= cost
            demand = self.state.market_demand * self.state.product_quality
            gain = int(80 + demand * 200 + random.gauss(0, 20))
            self.state.users += max(0, gain)

        elif action == "do_nothing":
            self.state.money += 0

        elif action == "pivot_strategy":
            cost = 11000
            self.state.money -= cost
            self.state.market_demand = min(1.0, self.state.market_demand + 0.12)
            self.state.product_quality = max(0.2, self.state.product_quality - 0.08)

    def _apply_environment_dynamics(self) -> None:
        self.state.burn_rate = max(6000, 8000 + self.state.team_size * 2200)
        self.state.money -= self.state.burn_rate * 0.08

        churn = int(self.state.users * max(0.01, 0.12 - self.state.product_quality * 0.08))
        self.state.users = max(0, self.state.users - churn)

        demand_change = random.uniform(-0.03, 0.04)
        self.state.market_demand = min(1.0, max(0.2, self.state.market_demand + demand_change))

        if self.state.money < 0:
            self.state.market_demand = max(0.0, self.state.market_demand - 0.1)

    def _compute_reward(self) -> float:
        revenue_growth = max(0, self.state.users - self.state.last_users) * self.state.product_quality
        user_growth = max(0, self.state.users - self.state.last_users)
        quality = self.state.product_quality
        burn = self.state.burn_rate / 1000.0

        reward = 1.2 * revenue_growth + 0.8 * user_growth + 2.0 * quality - 0.5 * burn
        if self.state.money < 0:
            reward -= 20.0
        return round(reward, 3)

    def _is_done(self) -> bool:
        if self.state.day >= self.max_days:
            return True
        if self.state.money < -20000:
            return True
        return False
