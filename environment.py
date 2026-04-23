import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HiddenState:
    market_demand: float = 0.7
    competition_level: float = 0.35
    economic_condition: float = 0.75


@dataclass
class PendingEffect:
    name: str
    eta: int
    payload: Dict[str, float]


@dataclass
class StartupState:
    day: int = 1
    money: float = 100000.0
    users: int = 150
    product_quality: float = 0.5
    team_size: int = 2
    burn_rate: float = 12000.0
    recent_actions: List[str] = field(default_factory=list)
    last_users: int = 150
    last_money: float = 100000.0
    last_event: str = "none"
    last_ad_performance: str = "average"
    recent_events: List[str] = field(default_factory=list)
    ignored_negative_trends: int = 0


class StartupEnvironment:
    ACTIONS = [
        "hire_employee",
        "fire_employee",
        "invest_in_product",
        "run_marketing_campaign",
        "do_nothing",
        "pivot_strategy",
    ]

    EVENT_TYPES = [
        "competitor_launch",
        "market_crash",
        "viral_growth",
        "tech_failure",
    ]

    def __init__(self, max_days: int = 30, seed: Optional[int] = None):
        self.max_days = max_days
        self.random = random.Random(seed)
        self.state = StartupState()
        self.hidden_state = HiddenState()
        self.pending_effects: List[PendingEffect] = []
        self.history: List[Dict[str, object]] = []
        self.reset()

    def reset(self) -> Dict[str, object]:
        self.state = StartupState()
        self.hidden_state = HiddenState()
        self.pending_effects = []
        self.history = []
        return self.get_observation()

    def get_observation(self) -> Dict[str, object]:
        recent_user_growth = self.state.users - self.state.last_users
        noisy_growth = recent_user_growth + self.random.randint(-6, 6)
        noisy_runway = self.state.money / max(self.state.burn_rate, 1)

        return {
            "day": self.state.day,
            "money": round(self.state.money, 2),
            "users": self.state.users,
            "product_quality": round(self.state.product_quality, 3),
            "team_size": self.state.team_size,
            "burn_rate": round(self.state.burn_rate, 2),
            "recent_user_growth": noisy_growth,
            "ad_performance": self.state.last_ad_performance,
            "recent_actions": list(self.state.recent_actions[-5:]),
            "recent_events": list(self.state.recent_events[-3:]),
            "runway_hint": round(noisy_runway + self.random.uniform(-0.5, 0.5), 2),
        }

    def get_debug_state(self) -> Dict[str, object]:
        return {
            "public_state": {
                "day": self.state.day,
                "money": round(self.state.money, 2),
                "users": self.state.users,
                "product_quality": round(self.state.product_quality, 3),
                "team_size": self.state.team_size,
                "burn_rate": round(self.state.burn_rate, 2),
                "recent_actions": list(self.state.recent_actions[-5:]),
                "recent_events": list(self.state.recent_events[-3:]),
            },
            "hidden_state": {
                "market_demand": round(self.hidden_state.market_demand, 3),
                "competition_level": round(self.hidden_state.competition_level, 3),
                "economic_condition": round(self.hidden_state.economic_condition, 3),
            },
            "pending_effects": [
                {"name": effect.name, "eta": effect.eta, "payload": effect.payload}
                for effect in self.pending_effects
            ],
        }

    def step(self, action: str, proposals: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.state.last_users = self.state.users
        self.state.last_money = self.state.money
        self.state.last_event = "none"

        self._apply_pending_effects()
        self._apply_action(action)
        self._apply_environment_dynamics(action)
        event = self._sample_event()
        reward = self._compute_reward(action)
        agent_rewards = self._compute_agent_rewards(action, reward, proposals or {})
        done = self._is_done()

        self.state.day += 1
        self.state.recent_actions.append(action)
        if len(self.state.recent_actions) > 10:
            self.state.recent_actions = self.state.recent_actions[-10:]

        observation = self.get_observation()
        debug_state = self.get_debug_state()

        step_record = {
            "day": self.state.day - 1,
            "action": action,
            "reward": reward,
            "event": event,
            "money": observation["money"],
            "users": observation["users"],
            "quality": observation["product_quality"],
        }
        self.history.append(step_record)

        return {
            "state": observation,
            "reward": reward,
            "done": done,
            "action": action,
            "event": event,
            "agent_rewards": agent_rewards,
            "debug_state": debug_state,
        }

    def _apply_pending_effects(self) -> None:
        remaining_effects: List[PendingEffect] = []
        for effect in self.pending_effects:
            effect.eta -= 1
            if effect.eta > 0:
                remaining_effects.append(effect)
                continue

            if effect.name == "hire_productivity_boost":
                self.state.product_quality = min(
                    1.0,
                    self.state.product_quality + effect.payload.get("quality_gain", 0.0),
                )
            elif effect.name == "product_quality_boost":
                self.state.product_quality = min(
                    1.0,
                    self.state.product_quality + effect.payload.get("quality_gain", 0.0),
                )
            elif effect.name == "marketing_decay":
                decay = int(effect.payload.get("user_decay", 0))
                self.state.users = max(0, self.state.users - decay)
            elif effect.name == "pivot_demand_shift":
                self.hidden_state.market_demand = min(
                    1.0,
                    max(0.0, self.hidden_state.market_demand + effect.payload.get("demand_gain", 0.0)),
                )

        self.pending_effects = remaining_effects

    def _apply_action(self, action: str) -> None:
        if action == "hire_employee":
            self.state.money -= 12000
            self.state.team_size += 1
            self.state.burn_rate += 2500
            self.pending_effects.append(
                PendingEffect(
                    name="hire_productivity_boost",
                    eta=2,
                    payload={"quality_gain": 0.04},
                )
            )

        elif action == "fire_employee":
            if self.state.team_size > 1:
                self.state.team_size -= 1
                self.state.money += 4000
                self.state.burn_rate = max(6000, self.state.burn_rate - 2000)
                self.state.product_quality = max(0.2, self.state.product_quality - 0.02)

        elif action == "invest_in_product":
            self.state.money -= 9000
            self.state.product_quality = min(1.0, self.state.product_quality + 0.03)
            self.pending_effects.extend(
                [
                    PendingEffect("product_quality_boost", 1, {"quality_gain": 0.03}),
                    PendingEffect("product_quality_boost", 3, {"quality_gain": 0.04}),
                ]
            )

        elif action == "run_marketing_campaign":
            self.state.money -= 7000
            success_prob = (
                0.2
                + 0.5 * self.hidden_state.market_demand
                + 0.2 * self.hidden_state.economic_condition
                + 0.15 * self.state.product_quality
                - 0.35 * self.hidden_state.competition_level
            )
            success_prob = max(0.05, min(0.95, success_prob))

            if self.random.random() < success_prob:
                gain = self.random.randint(90, 180)
                self.state.last_ad_performance = "good"
            else:
                gain = self.random.randint(5, 60)
                self.state.last_ad_performance = "poor" if gain < 20 else "average"

            self.state.users += gain
            self.pending_effects.append(
                PendingEffect(
                    name="marketing_decay",
                    eta=2,
                    payload={"user_decay": max(5, int(gain * 0.18))},
                )
            )

        elif action == "do_nothing":
            self.state.last_ad_performance = "average"

        elif action == "pivot_strategy":
            self.state.money -= 11000
            self.state.product_quality = max(0.25, self.state.product_quality - 0.05)
            self.pending_effects.append(
                PendingEffect(
                    name="pivot_demand_shift",
                    eta=2,
                    payload={"demand_gain": 0.14},
                )
            )

    def _apply_environment_dynamics(self, action: str) -> None:
        recurring_revenue = self.state.users * (0.22 + 0.35 * self.state.product_quality)
        recurring_revenue *= 0.7 + 0.3 * self.hidden_state.economic_condition
        self.state.money += recurring_revenue

        self.state.burn_rate = max(7000, 7500 + self.state.team_size * 2200)
        self.state.money -= self.state.burn_rate * 0.09

        churn_rate = (
            0.09
            - 0.05 * self.state.product_quality
            + 0.04 * self.hidden_state.competition_level
            - 0.025 * self.hidden_state.economic_condition
        )
        churn = int(self.state.users * max(0.01, churn_rate))
        self.state.users = max(0, self.state.users - churn)

        self.hidden_state.market_demand = min(
            1.0,
            max(0.05, self.hidden_state.market_demand + self.random.uniform(-0.04, 0.05)),
        )
        self.hidden_state.competition_level = min(
            1.0,
            max(0.05, self.hidden_state.competition_level + self.random.uniform(-0.03, 0.04)),
        )
        self.hidden_state.economic_condition = min(
            1.0,
            max(0.2, self.hidden_state.economic_condition + self.random.uniform(-0.03, 0.03)),
        )

        recent_growth = self.state.users - self.state.last_users
        if recent_growth < 0 and action in {"do_nothing", "hire_employee"}:
            self.state.ignored_negative_trends += 1
        else:
            self.state.ignored_negative_trends = max(0, self.state.ignored_negative_trends - 1)

    def _sample_event(self) -> str:
        event_probability = 0.22
        if self.random.random() > event_probability:
            self.state.recent_events.append("none")
            self.state.recent_events = self.state.recent_events[-5:]
            return "none"

        event = self.random.choice(self.EVENT_TYPES)
        self.state.last_event = event

        if event == "competitor_launch":
            loss = self.random.randint(15, 50)
            self.state.users = max(0, self.state.users - loss)
            self.hidden_state.competition_level = min(1.0, self.hidden_state.competition_level + 0.12)
        elif event == "market_crash":
            self.hidden_state.market_demand = max(0.05, self.hidden_state.market_demand - 0.18)
            self.hidden_state.economic_condition = max(0.2, self.hidden_state.economic_condition - 0.15)
            self.state.money -= 2500
        elif event == "viral_growth":
            gain = self.random.randint(50, 120)
            self.state.users += gain
            self.hidden_state.market_demand = min(1.0, self.hidden_state.market_demand + 0.08)
        elif event == "tech_failure":
            self.state.product_quality = max(0.15, self.state.product_quality - 0.12)
            self.state.money -= 1800

        self.state.recent_events.append(event)
        self.state.recent_events = self.state.recent_events[-5:]
        return event

    def _compute_reward(self, action: str) -> float:
        long_term_user_growth = self.state.users - self.state.last_users
        profit = self.state.money - self.state.last_money
        product_quality = self.state.product_quality
        burn = self.state.burn_rate / 1000.0

        instability_penalty = 0.0
        if len(self.state.recent_actions) >= 2:
            last_two = self.state.recent_actions[-2:]
            if len(set(last_two + [action])) == 3:
                instability_penalty += 2.0
            if last_two[-1] != action:
                instability_penalty += 0.75

        negative_trend_penalty = 1.5 * self.state.ignored_negative_trends
        bankruptcy_penalty = 25.0 if self.state.money < 0 else 0.0

        reward = (
            0.4 * long_term_user_growth
            + 0.3 * (profit / 1000.0)
            + 12.0 * product_quality
            - 0.2 * burn
            - instability_penalty
            - negative_trend_penalty
            - bankruptcy_penalty
        )
        return round(reward, 3)

    def _compute_agent_rewards(
        self,
        chosen_action: str,
        reward: float,
        proposals: Dict[str, object],
    ) -> Dict[str, float]:
        agent_rewards = {"CEO": reward}
        for agent_name, proposal in proposals.items():
            aligned = getattr(proposal, "action", None) == chosen_action
            agent_rewards[agent_name] = round(reward if aligned else reward * 0.2, 3)
        return agent_rewards

    def _is_done(self) -> bool:
        if self.state.day >= self.max_days:
            return True
        if self.state.money < -25000:
            return True
        if self.state.users <= 0:
            return True
        return False
