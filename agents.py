import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ActionProposal:
    action: str
    reasoning: str


class BaseCoFounder:
    def __init__(self, name: str):
        self.name = name

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        raise NotImplementedError

    @staticmethod
    def _maybe_break_loop(
        observation: Dict[str, object],
        preferred_action: str,
        fallback_action: str,
        preferred_reason: str,
        fallback_reason: str,
    ) -> ActionProposal:
        repeated_action = observation.get("last_action") == preferred_action
        streak = observation.get("consecutive_action_streak", 0)
        if repeated_action and streak >= 2:
            return ActionProposal(action=fallback_action, reasoning=fallback_reason)
        return ActionProposal(action=preferred_action, reasoning=preferred_reason)


def _average_growth(observation: Dict[str, object]) -> float:
    growth_window = observation.get("last_3_growth", [])
    if not growth_window:
        return float(observation.get("recent_user_growth", 0))
    return sum(growth_window) / len(growth_window)


def _has_event(observation: Dict[str, object], event_name: str) -> bool:
    return event_name in observation.get("recent_events", [])


def _is_crisis(observation: Dict[str, object]) -> bool:
    return bool(observation.get("is_crisis", False))


def _recently_pivoted(observation: Dict[str, object]) -> bool:
    return "pivot_strategy" in observation.get("recent_actions", [])[-2:]


def _is_crisis_unsafe_action(action: str) -> bool:
    return action in {"run_marketing_campaign", "hire_employee"}


def _last_3_growth(observation: Dict[str, object]) -> Tuple[float, ...]:
    return tuple(observation.get("last_3_growth", ()))


def _growth_consistently_positive(observation: Dict[str, object]) -> bool:
    growth_window = _last_3_growth(observation)
    return len(growth_window) >= 3 and all(growth > 0 for growth in growth_window)


def _growth_strongly_negative(observation: Dict[str, object]) -> bool:
    growth_window = _last_3_growth(observation)
    return bool(growth_window) and sum(growth_window) / len(growth_window) < -15


def _negative_growth_streak(observation: Dict[str, object]) -> int:
    streak = 0
    for growth in reversed(_last_3_growth(observation)):
        if growth >= 0:
            break
        streak += 1
    return streak


def _mostly_negative_growth(observation: Dict[str, object]) -> bool:
    growth_window = _last_3_growth(observation)
    if not growth_window:
        return observation.get("recent_user_growth", 0) < 0
    negative_count = sum(1 for growth in growth_window if growth < 0)
    return negative_count >= 2


def _action_repeated_twice(observation: Dict[str, object], action: str) -> bool:
    return observation.get("last_action") == action and observation.get("consecutive_action_streak", 0) >= 2


def _recently_invested_in_product(observation: Dict[str, object]) -> bool:
    return "invest_in_product" in observation.get("recent_actions", [])[-2:]


class TechCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Tech Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        quality = observation["product_quality"]
        trend = observation["trend_direction"]
        average_growth = _average_growth(observation)
        runway_hint = observation["runway_hint"]
        money = observation["money"]
        last_action = observation.get("last_action")

        if _is_crisis(observation):
            if _recently_pivoted(observation):
                if quality < 0.62 and money >= 6000:
                    return ActionProposal(
                        action="invest_in_product",
                        reasoning=(
                            "We already made a strategic reset recently, so the next crisis move should improve the product behind that reset."
                        ),
                    )
                return ActionProposal(
                    action="do_nothing",
                    reasoning=(
                        "We already made a strategic reset recently, so I would avoid another spend-heavy move and preserve runway for the next signal."
                    ),
                )
            if (_has_event(observation, "tech_failure") or quality < 0.46) and money >= 9000:
                return ActionProposal(
                    action="invest_in_product",
                    reasoning=(
                        "We are in crisis mode, and the product looks weak enough that recovery will fail unless we quickly repair core quality."
                    ),
                )
            return ActionProposal(
                action="pivot_strategy",
                reasoning=(
                    "We are in crisis mode, so freezing would be fatal. I would change direction fast and look for a more responsive market position."
                ),
            )

        if _has_event(observation, "tech_failure") or quality < 0.58:
            return self._maybe_break_loop(
                observation,
                preferred_action="invest_in_product",
                fallback_action="pivot_strategy",
                preferred_reason=(
                    "Signals suggest product stability is at risk, so we should improve quality before scaling further."
                ),
                fallback_reason=(
                    "We have already repeated product work several turns, so I would test a strategic reset instead of pausing and losing more ground."
                ),
            )

        if trend == "declining" and average_growth < -10:
            if runway_hint < 1.7:
                return ActionProposal(
                    action="do_nothing",
                    reasoning=(
                        "Growth is weakening, but runway is tight enough that I would avoid another spend-heavy move until we get a clearer signal."
                    ),
                )
            return self._maybe_break_loop(
                observation,
                preferred_action="invest_in_product",
                fallback_action="pivot_strategy",
                preferred_reason=(
                    "The growth trend is declining, and quality is not strong enough to confidently absorb churn, so product improvement is the safest bet."
                ),
                fallback_reason=(
                    "We keep returning to the same product action, so a small strategic reset may reveal whether the problem is positioning rather than execution."
                ),
            )

        if trend == "improving" and quality >= 0.72 and runway_hint > 3.0:
            return ActionProposal(
                action="hire_employee",
                reasoning=(
                    "Trend signals are improving and product quality looks solid enough that adding capacity could compound momentum."
                ),
            )

        return ActionProposal(
            action="invest_in_product",
            reasoning=(
                "The situation is still uncertain, but steady product investment should improve retention and make future growth more durable."
            ),
        )


class GrowthCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Growth Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        ad_performance = observation["ad_performance"]
        trend = observation["trend_direction"]
        average_growth = _average_growth(observation)
        runway_hint = observation["runway_hint"]
        quality = observation["product_quality"]
        last_action = observation.get("last_action")

        if _is_crisis(observation):
            if _recently_pivoted(observation):
                if ad_performance == "poor" and quality < 0.62:
                    return ActionProposal(
                        action="invest_in_product",
                        reasoning=(
                            "We already pivoted recently and demand still looks weak, so the next recovery attempt should strengthen the product before another reset."
                        ),
                    )
                return ActionProposal(
                    action="do_nothing",
                    reasoning=(
                        "We already changed direction recently, so I would preserve cash and wait for the pivot signal instead of buying growth during crisis."
                    ),
                )
            if ad_performance != "poor" and quality >= 0.52 and observation["money"] >= 7000:
                return ActionProposal(
                    action="pivot_strategy",
                    reasoning=(
                        "We are in crisis mode, so I would focus on a cheaper strategic reset instead of spending scarce cash on acquisition."
                    ),
                )
            return ActionProposal(
                action="pivot_strategy",
                reasoning=(
                    "We are in crisis mode and current growth is not reliable enough to wait, so I would force a reset in positioning rather than freeze."
                ),
            )

        if _has_event(observation, "viral_growth") and runway_hint > 1.5:
            return self._maybe_break_loop(
                observation,
                preferred_action="run_marketing_campaign",
                fallback_action="hire_employee",
                preferred_reason=(
                    "Momentum is already positive, so leaning into acquisition could convert attention into a larger user base."
                ),
                fallback_reason=(
                    "We have already repeated the same growth motion, so I would support the momentum with extra capacity instead of another campaign right away."
                ),
            )

        if runway_hint < 1.3:
            return ActionProposal(
                action="do_nothing",
                reasoning=(
                    "Growth signals are noisy and runway is too short to confidently spend on acquisition this turn."
                ),
            )

        if trend == "declining" and (ad_performance == "poor" or average_growth < -12):
            return self._maybe_break_loop(
                observation,
                preferred_action="pivot_strategy",
                fallback_action="invest_in_product",
                preferred_reason=(
                    "Recent growth signals are worsening, so I suspect the current positioning or demand fit is soft and a pivot is worth testing."
                ),
                fallback_reason=(
                    "We have already repeated strategic resets, so I would strengthen the product instead of pivoting again without better evidence."
                ),
            )

        if ad_performance != "poor" and (trend == "improving" or observation["users"] < 450 or average_growth < 35):
            return self._maybe_break_loop(
                observation,
                preferred_action="run_marketing_campaign",
                fallback_action="pivot_strategy",
                preferred_reason=(
                    "Demand signals remain usable, and the recent trend does not argue for retreat, so I would keep trying to grow users."
                ),
                fallback_reason=(
                    "Marketing has been repeated several turns, so I would change direction rather than keep buying similar signals."
                ),
            )

        return ActionProposal(
            action="pivot_strategy",
            reasoning=(
                "The current growth picture is mixed, and under uncertainty I would rather test a repositioning than keep spending on a weak channel."
            ),
        )


class FinanceCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Finance Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        money = observation["money"]
        burn_rate = observation["burn_rate"]
        runway_hint = observation["runway_hint"]
        trend = observation["trend_direction"]
        average_growth = _average_growth(observation)
        team_size = observation["team_size"]
        ad_performance = observation["ad_performance"]
        last_action = observation.get("last_action")

        if _is_crisis(observation):
            if team_size > 2:
                return ActionProposal(
                    action="fire_employee",
                    reasoning=(
                        "We are in crisis mode, so the first job is buying survival time by cutting burn immediately."
                    ),
                )
            if _recently_pivoted(observation):
                if ad_performance != "poor":
                    return ActionProposal(
                        action="do_nothing",
                        reasoning=(
                            "We already reset strategy recently, so I would preserve cash instead of funding a growth push before the reset has time to show evidence."
                        ),
                    )
                return ActionProposal(
                    action="invest_in_product",
                    reasoning=(
                        "We already reset strategy recently, and demand is still weak, so I would invest in the product rather than spend another turn pivoting."
                    ),
                )
            if average_growth > -5 and ad_performance != "poor" and money >= 7000:
                return ActionProposal(
                    action="do_nothing",
                    reasoning=(
                        "The team is already lean, so I would preserve cash and avoid a marketing spend while the company is in crisis."
                    ),
                )
            return ActionProposal(
                action="pivot_strategy",
                reasoning=(
                    "The company is already lean and under acute pressure, so I would attempt a strategic reset instead of sitting still."
                ),
            )

        if money < 25000 or runway_hint < 1.4:
            if team_size <= 2 or (observation.get("last_action") == "fire_employee" and observation.get("consecutive_action_streak", 0) >= 2):
                return ActionProposal(
                    action="pivot_strategy",
                    reasoning=(
                        "Runway is tight, and if we cannot responsibly cut more headcount then we should change strategy rather than drift."
                    ),
                )
            return ActionProposal(
                action="fire_employee",
                reasoning=(
                    "Cash runway looks critically tight, so reducing burn is the clearest survival move even though the signal is imperfect."
                ),
            )

        if burn_rate > 18000 and trend == "declining":
            return ActionProposal(
                action="do_nothing",
                reasoning=(
                    "The company is spending heavily while trend signals soften, so I would protect cash until we learn more."
                ),
            )

        if runway_hint > 4.0 and trend == "improving" and average_growth > 15:
            return ActionProposal(
                action="hire_employee",
                reasoning=(
                    "Runway is healthy and growth looks stronger than before, so measured team expansion is financially supportable."
                ),
            )

        return ActionProposal(
            action="do_nothing",
            reasoning=(
                "The financial picture is acceptable but uncertain, so I would avoid a sharp move and wait for clearer evidence from the next trend update."
            ),
        )


class CEO:
    def __init__(self):
        self.name = "CEO"

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        if not proposals:
            proposals = {}

        focus, focus_reason = self._determine_focus(observation)
        action, mode, trigger, choice_reason, rejected = self._choose_strict_policy_action(observation)
        action, rejected, override_reason = self._apply_strict_anti_patterns(action, observation, rejected)
        if override_reason:
            choice_reason = override_reason
        matching_proposal = self._find_matching_proposal(proposals, action)
        proposal_context = (
            f" Supporting proposal: {matching_proposal.reasoning}"
            if matching_proposal is not None
            else " No co-founder proposal matched the strict CEO policy, so the CEO synthesized the action."
        )

        return ActionProposal(
            action=action,
            reasoning=(
                f"Mode: {mode}. "
                f"Trigger: {trigger}. "
                f"Selected action: {action}. {choice_reason}. "
                f"Rejected options: {rejected}. "
                f"Checked runway={observation['runway_hint']}, last_3_growth={observation['last_3_growth']}, "
                f"recent_events={observation['recent_events']}. "
                f"Policy focus: {focus} ({focus_reason})."
                f"{proposal_context}"
            ),
        )

        best_agent_name = None
        best_proposal = None
        best_score = float("-inf")

        for agent_name, proposal in proposals.items():
            score = self._score_proposal(agent_name, proposal, observation, focus)
            if score > best_score:
                best_agent_name = agent_name
                best_proposal = proposal
                best_score = score

        if best_proposal is None:
            best_agent_name, best_proposal = random.choice(list(proposals.items()))
        elif observation.get("is_crisis") and best_proposal.action == "do_nothing":
            best_agent_name, best_proposal = max(
                (
                    (name, proposal)
                    for name, proposal in proposals.items()
                    if proposal.action != "do_nothing"
                ),
                default=(best_agent_name, best_proposal),
                key=lambda item: self._score_proposal(item[0], item[1], observation, focus),
            )
        elif best_proposal.action == "pivot_strategy" and _recently_pivoted(observation):
            best_agent_name, best_proposal = max(
                (
                    (name, proposal)
                    for name, proposal in proposals.items()
                    if proposal.action != "pivot_strategy"
                ),
                default=(best_agent_name, best_proposal),
                key=lambda item: self._score_proposal(item[0], item[1], observation, focus),
            )

        return ActionProposal(
            action=best_proposal.action,
            reasoning=(
                f"CEO prioritized {focus} because {focus_reason}. "
                f"Crisis status is {observation.get('crisis_level', 'unknown')} ({observation.get('crisis_reason', 'no crisis reason')}). "
                f"Recent trend is {observation['trend_direction']} with growth history {observation['last_3_growth']}. "
                f"Selected {best_proposal.action} from {best_agent_name} because it best fits the current trade-off under uncertainty. "
                f"Co-founder rationale: {best_proposal.reasoning}"
            ),
        )

    def _choose_strict_policy_action(self, observation: Dict[str, object]) -> Tuple[str, str, str, str, str]:
        runway = observation["runway_hint"]
        growth = observation.get("recent_user_growth", 0)
        trend = observation["trend_direction"]
        negative_streak = _negative_growth_streak(observation)
        mostly_negative = _mostly_negative_growth(observation)
        team_size = observation["team_size"]
        recent_actions = observation.get("recent_actions", [])
        last_action = observation.get("last_action")

        if runway < 2.0:
            if team_size > 1 and not _action_repeated_twice(observation, "fire_employee"):
                return (
                    "fire_employee",
                    "Survival",
                    "runway is below 2, so burn reduction is mandatory",
                    "fire_employee is the fastest available burn reduction and takes priority over recovery or optimization",
                    "marketing and hiring are forbidden; do_nothing does not reduce burn; pivot is secondary to immediate burn reduction",
                )
            if not (last_action == "do_nothing" and observation.get("consecutive_action_streak", 0) >= 2):
                return (
                    "do_nothing",
                    "Survival",
                    "runway is below 2 and further firing is unavailable or already repeated",
                    "do_nothing preserves cash without adding burn while the company is in survival mode",
                    "marketing and hiring are forbidden; pivot is avoided while cash is critically low unless inaction has already repeated",
                )
            return (
                "pivot_strategy",
                "Survival",
                "runway is below 2 and do_nothing has already repeated",
                "pivot_strategy is the remaining reversible intervention after burn reduction and cash preservation are exhausted",
                "marketing and hiring are forbidden; repeated do_nothing during crisis is forbidden; firing is unavailable or already repeated",
            )

        if negative_streak >= 3:
            if not _recently_invested_in_product(observation):
                return (
                    "invest_in_product",
                    "Recovery",
                    "growth has been negative for 3 consecutive turns",
                    "invest_in_product is the default recovery intervention when the company has not recently repaired the product",
                    "do_nothing is forbidden during decline; marketing is risky before retention improves; pivot is reserved for decline after product investment",
                )
            return (
                "pivot_strategy",
                "Recovery",
                "growth has stayed negative after recent product investment",
                "pivot_strategy is the next recovery move because product intervention has already been tried recently",
                "do_nothing is forbidden during decline; marketing and hiring are rejected; product was already tried recently, so pivot is preferred unless anti-freeze blocks it",
            )

        if mostly_negative or trend == "declining":
            if _recently_invested_in_product(observation) or _action_repeated_twice(observation, "invest_in_product"):
                return (
                    "pivot_strategy",
                "Recovery",
                "last_3_growth is mostly negative or users are decreasing",
                "pivot_strategy is justified because decline is continuing after recent or repeated product work",
                "do_nothing is forbidden during decline; hiring increases burn; marketing can amplify a leaky product; more product work is lower priority unless anti-freeze blocks pivot",
            )
            return (
                "invest_in_product",
                "Recovery",
                "last_3_growth is mostly negative or users are decreasing",
                "invest_in_product is the default intervention to improve retention and product quality before chasing growth",
                "do_nothing is forbidden during decline; pivot waits until product intervention has been tried; hiring increases burn",
            )

        if runway < 3.0:
            if team_size > 1 and not _action_repeated_twice(observation, "fire_employee"):
                return (
                    "fire_employee",
                    "Survival",
                    "runway is below 3, so cash preservation dominates",
                    "fire_employee extends runway and avoids adding burn",
                    "hiring is forbidden; marketing is too cash-intensive; pivot is avoided without strong decline",
                )
            return (
                "invest_in_product",
                "Optimization",
                "runway is below 3 but immediate firing is unavailable or already repeated",
                "invest_in_product is the safest active default because do_nothing should not be the default",
                "hiring is forbidden; marketing is cautious under low runway; pivot is reserved for sustained decline",
            )

        if growth > 50 and trend == "improving" and runway > 3.0:
            return (
                "run_marketing_campaign",
                "Optimization",
                "growth is strongly positive and trend is improving",
                "run_marketing_campaign captures momentum without increasing fixed costs",
                "hiring is rejected because fixed costs should not rise on short-term signals; do_nothing wastes momentum; pivot is unnecessary",
            )

        if 3.0 <= runway <= 6.0:
            return (
                "invest_in_product",
                "Optimization",
                "runway is mid-stage, so efficiency matters",
                "invest_in_product improves quality without permanent burn increase",
                "hiring is forbidden in mid-stage runway; marketing is selective and only used for strong improving growth; do_nothing is not the default",
            )

        if runway > 6.0:
            if (
                _growth_consistently_positive(observation)
                and "viral_growth" not in observation.get("recent_events", [])
                and not _action_repeated_twice(observation, "hire_employee")
            ):
                return (
                    "hire_employee",
                    "Optimization",
                    "runway is healthy and all last 3 growth values are positive without a recent viral spike",
                    "hire_employee is allowed rarely because the company has enough runway and stable growth",
                    "do_nothing is not the default; pivot is unnecessary; product and marketing remain valid but capacity can now compound execution",
                )
            if recent_actions[-2:] == ["invest_in_product", "invest_in_product"]:
                return (
                    "run_marketing_campaign",
                    "Optimization",
                    "runway is healthy and product investment has already repeated twice",
                    "run_marketing_campaign balances growth and product work without adding fixed costs",
                    "repeating invest_in_product is forbidden; hiring is rejected unless growth is consistently positive and not viral-driven",
                )
            return (
                "invest_in_product",
                "Optimization",
                "no stronger survival or recovery rule applies",
                "invest_in_product is the default active choice and avoids passive drift",
                "do_nothing is not the default; hiring is rare; pivot waits for sustained decline",
            )

        return (
            "invest_in_product",
            "Optimization",
            "no stronger rule applies",
            "invest_in_product is the default active choice",
            "do_nothing is not the default; hiring and marketing require stronger evidence",
        )

    def _apply_strict_anti_patterns(
        self,
        action: str,
        observation: Dict[str, object],
        rejected: str,
    ) -> Tuple[str, str, str]:
        growth = observation.get("recent_user_growth", 0)
        last_action = observation.get("last_action")
        runway = observation["runway_hint"]
        trend = observation["trend_direction"]

        if action == "do_nothing":
            do_nothing_allowed = (
                growth >= 0
                and last_action != "do_nothing"
                and not _mostly_negative_growth(observation)
                and trend != "declining"
            )
            if not do_nothing_allowed:
                if runway < 2.0:
                    replacement = "fire_employee" if observation["team_size"] > 1 else "pivot_strategy"
                else:
                    replacement = "fire_employee" if runway < 3.0 and observation["team_size"] > 1 else "invest_in_product"
                return (
                    replacement,
                    f"{rejected}; do_nothing is forbidden because growth is negative, repeated, or decline is sustained",
                    f"{replacement} is the safest allowed replacement because do_nothing is forbidden by the anti-freeze rule",
                )

        if _action_repeated_twice(observation, action):
            replacement = self._best_non_repeated_action(action, observation)
            return (
                replacement,
                f"{rejected}; {action} was already repeated twice, so repetition control forced {replacement}",
                f"{replacement} is the best non-repeated action after {action} hit the two-turn repetition limit",
            )

        if last_action == action and growth < 0:
            replacement = self._best_non_repeated_action(action, observation)
            return (
                replacement,
                f"{rejected}; the last {action} led into negative growth, so repetition control forced {replacement}",
                f"{replacement} avoids repeating {action} after it led into negative growth",
            )

        return action, rejected, ""

    def _best_non_repeated_action(self, action: str, observation: Dict[str, object]) -> str:
        runway = observation["runway_hint"]
        if runway < 2.0:
            for candidate in ("fire_employee", "do_nothing", "pivot_strategy"):
                if candidate != action:
                    return candidate
        if _mostly_negative_growth(observation) or observation["trend_direction"] == "declining":
            for candidate in ("invest_in_product", "pivot_strategy"):
                if candidate != action:
                    return candidate
        for candidate in ("invest_in_product", "run_marketing_campaign", "pivot_strategy", "do_nothing"):
            if candidate != action:
                return candidate
        return "invest_in_product"

    @staticmethod
    def _find_matching_proposal(
        proposals: Dict[str, ActionProposal],
        action: str,
    ) -> ActionProposal | None:
        for proposal in proposals.values():
            if proposal.action == action:
                return proposal
        return None

    def _choose_by_runway_policy(
        self,
        proposals: Dict[str, ActionProposal],
        observation: Dict[str, object],
        focus: str,
    ) -> Tuple[str, ActionProposal, str] | None:
        runway = observation["runway_hint"]
        trend = observation["trend_direction"]
        recent_events = observation["recent_events"]
        recent_actions = observation.get("recent_actions", [])
        strongly_negative = _growth_strongly_negative(observation)
        consistently_positive = _growth_consistently_positive(observation)

        if runway < 2.0:
            if observation["team_size"] > 1:
                return (
                    "CEO runway policy",
                    ActionProposal(
                        action="fire_employee",
                        reasoning="Runway is below 2 and the team can still be reduced, so the CEO selected the top survival action.",
                    ),
                    "Runway is below 2, so survival policy overrides growth and fixed-cost actions.",
                )
            allowed_actions = {"fire_employee", "do_nothing", "pivot_strategy"}
            priority = ["fire_employee", "do_nothing", "pivot_strategy"]
            if observation.get("last_action") == "do_nothing" and observation.get("consecutive_action_streak", 0) >= 2:
                allowed_actions.discard("do_nothing")
                priority = ["fire_employee", "pivot_strategy"]
            return self._choose_from_priority(
                proposals,
                observation,
                focus,
                priority=priority,
                allowed_actions=allowed_actions,
                reason="Runway is below 2, so survival policy overrides growth and fixed-cost actions.",
            )

        if 2.0 <= runway < 3.0:
            if observation["team_size"] > 1:
                return (
                    "CEO runway policy",
                    ActionProposal(
                        action="fire_employee",
                        reasoning="Runway is in the warning zone and the team can still be reduced, so the CEO selected the strongest cash-preservation action.",
                    ),
                    "Runway is in the warning zone, so the CEO strongly prioritizes burn reduction.",
                )
            allowed_actions = {"fire_employee", "do_nothing", "invest_in_product"}
            priority = ["fire_employee", "do_nothing", "invest_in_product"]
            if strongly_negative:
                allowed_actions.add("pivot_strategy")
                priority.append("pivot_strategy")
            return self._choose_from_priority(
                proposals,
                observation,
                focus,
                priority=priority,
                allowed_actions=allowed_actions,
                reason="Runway is in the warning zone, so the CEO avoids hiring and only pivots on strongly negative growth.",
            )

        if 3.0 <= runway <= 6.0:
            allowed_actions = {"fire_employee", "do_nothing", "invest_in_product", "pivot_strategy"}
            if trend == "improving":
                allowed_actions.add("run_marketing_campaign")
            return self._choose_from_priority(
                proposals,
                observation,
                focus,
                priority=["do_nothing", "invest_in_product", "fire_employee", "pivot_strategy", "run_marketing_campaign"],
                allowed_actions=allowed_actions,
                reason="Runway is mid-stage, so the CEO avoids hiring and only funds marketing when the trend is improving.",
            )

        if runway > 6.0:
            allowed_actions = set(proposal.action for proposal in proposals.values())
            if not consistently_positive or "viral_growth" in recent_events:
                allowed_actions.discard("hire_employee")
            if recent_actions[-2:] == ["invest_in_product", "invest_in_product"]:
                allowed_actions.discard("invest_in_product")
            if not allowed_actions:
                allowed_actions = {"do_nothing"}
            return self._choose_from_priority(
                proposals,
                observation,
                focus,
                priority=["run_marketing_campaign", "invest_in_product", "do_nothing", "pivot_strategy", "hire_employee"],
                allowed_actions=allowed_actions,
                reason="Runway is healthy, so the CEO can balance product and growth while avoiding fixed costs after short-term spikes.",
            )

        return None

    def _choose_from_priority(
        self,
        proposals: Dict[str, ActionProposal],
        observation: Dict[str, object],
        focus: str,
        priority: list[str],
        allowed_actions: set[str],
        reason: str,
    ) -> Tuple[str, ActionProposal, str]:
        candidates = [
            (agent_name, proposal)
            for agent_name, proposal in proposals.items()
            if proposal.action in allowed_actions
        ]

        if not candidates:
            fallback_action = "fire_employee" if "fire_employee" in allowed_actions else "do_nothing"
            return (
                "CEO policy fallback",
                ActionProposal(
                    action=fallback_action,
                    reasoning="No co-founder proposal satisfied the runway policy, so the CEO selected the safest allowed fallback.",
                ),
                reason,
            )

        return (
            *max(
                candidates,
                key=lambda item: (
                    -priority.index(item[1].action) if item[1].action in priority else -len(priority),
                    self._score_proposal(item[0], item[1], observation, focus),
                ),
            ),
            reason,
        )

    def _choose_crisis_action(
        self,
        proposals: Dict[str, ActionProposal],
        observation: Dict[str, object],
        focus: str,
    ) -> Tuple[str, ActionProposal]:
        priority = ["fire_employee", "pivot_strategy", "do_nothing", "invest_in_product"]
        safe_proposals = [
            (agent_name, proposal)
            for agent_name, proposal in proposals.items()
            if not _is_crisis_unsafe_action(proposal.action)
        ]

        if not safe_proposals:
            return (
                "CEO safety override",
                ActionProposal(
                    action="fire_employee",
                    reasoning="Crisis constraints blocked all unsafe growth actions, so the CEO selected an immediate survival action.",
                ),
            )

        return max(
            safe_proposals,
            key=lambda item: (
                -priority.index(item[1].action) if item[1].action in priority else -len(priority),
                self._score_proposal(item[0], item[1], observation, focus),
            ),
        )

    def _determine_focus(self, observation: Dict[str, object]) -> Tuple[str, str]:
        runway_hint = observation["runway_hint"]
        trend = observation["trend_direction"]
        average_growth = _average_growth(observation)
        recent_events = observation["recent_events"]
        quality = observation["product_quality"]

        if observation.get("is_crisis"):
            if observation["money"] < 0:
                return "crisis_recovery", "the company is already out of cash and needs an immediate survival-and-recovery move"
            if runway_hint < 0.75:
                return "crisis_recovery", "runway is nearly gone, so waiting is more dangerous than a bold recovery attempt"
            return "crisis_recovery", "the company is in crisis and needs an active recovery move rather than caution"
        if runway_hint < 1.0:
            return "survival", "runway looks critically short and survival has to come before expansion"
        if trend == "declining" and average_growth < -12 and runway_hint < 2.0:
            return "survival", "growth is deteriorating while cash looks fragile"
        if "tech_failure" in recent_events or quality < 0.55:
            return "product_stabilization", "product risk appears elevated from recent signals"
        if trend == "declining" and average_growth < -12:
            return "reset", "the trend is clearly worsening and the current approach may not be working"
        if trend == "improving" and average_growth > 20 and runway_hint > 1.5:
            return "growth_capture", "recent growth looks positive enough to justify leaning into momentum"
        return "balanced", "signals are mixed, so the best move should preserve flexibility"

    def _score_proposal(
        self,
        agent_name: str,
        proposal: ActionProposal,
        observation: Dict[str, object],
        focus: str,
    ) -> float:
        action = proposal.action
        trend = observation["trend_direction"]
        average_growth = _average_growth(observation)
        runway_hint = observation["runway_hint"]
        quality = observation["product_quality"]
        streak = observation.get("consecutive_action_streak", 0)
        repeated_action = observation.get("last_action") == action

        score = 0.2

        if action == "fire_employee":
            score += 1.2 if runway_hint < 1.6 else 0.2
            score -= 0.7 if observation["team_size"] <= 2 else 0.0
        elif action == "do_nothing":
            score += 0.8 if runway_hint < 2.0 or observation["burn_rate"] > 18000 else 0.3
            score += 0.3 if trend == "stable" else 0.0
        elif action == "invest_in_product":
            score += 1.0 + (1.0 - quality) * 1.6
            score += 0.6 if trend == "declining" else 0.0
            score += 0.7 if _has_event(observation, "tech_failure") else 0.0
        elif action == "pivot_strategy":
            score += 0.9 if trend == "declining" else 0.3
            score += 0.5 if _has_event(observation, "competitor_launch") or _has_event(observation, "market_crash") else 0.0
            score -= 0.3 if runway_hint < 1.4 else 0.0
        elif action == "run_marketing_campaign":
            score += 0.8 if trend == "improving" else 0.2
            score += 0.5 if observation["ad_performance"] == "good" else 0.0
            score -= 0.8 if observation["ad_performance"] == "poor" else 0.0
            score -= 0.5 if runway_hint < 1.5 else 0.0
        elif action == "hire_employee":
            score += 0.8 if runway_hint > 3.5 else 0.2
            score += 0.4 if trend == "improving" and average_growth > 10 else 0.0
            score -= 0.8 if runway_hint < 2.0 else 0.0

        focus_bonus = {
            "crisis_recovery": {
                "fire_employee": 1.0,
                "pivot_strategy": 1.2,
                "run_marketing_campaign": 0.9,
                "invest_in_product": 0.8,
                "do_nothing": -2.6,
                "hire_employee": -1.4,
            },
            "survival": {"fire_employee": 1.0, "do_nothing": 0.7, "hire_employee": -1.0, "run_marketing_campaign": -0.6},
            "product_stabilization": {"invest_in_product": 0.9, "hire_employee": 0.2, "run_marketing_campaign": -0.4},
            "reset": {"pivot_strategy": 0.9, "invest_in_product": 0.5, "run_marketing_campaign": -0.5},
            "growth_capture": {"run_marketing_campaign": 0.9, "hire_employee": 0.4, "fire_employee": -0.8},
            "balanced": {"invest_in_product": 0.2, "do_nothing": 0.2, "pivot_strategy": 0.1},
        }
        score += focus_bonus.get(focus, {}).get(action, 0.0)

        if focus == "crisis_recovery":
            if _is_crisis_unsafe_action(action):
                score -= 10.0
            if action == "fire_employee" and observation["team_size"] <= 2:
                score -= 1.5
            if action == "pivot_strategy" and _recently_pivoted(observation):
                score -= 2.2
            if action == "pivot_strategy" and (trend == "declining" or _has_event(observation, "competitor_launch") or _has_event(observation, "market_crash")):
                score += 0.8
            if action == "invest_in_product" and _recently_pivoted(observation) and quality < 0.65:
                score += 0.7
            if action == "invest_in_product" and (_has_event(observation, "tech_failure") or quality < 0.45):
                score += 0.8
            if action == "do_nothing":
                score -= 2.0

        if repeated_action and streak >= 2:
            score -= 1.1 + 0.25 * streak

        if focus == "crisis_recovery" and action == "fire_employee":
            score += 0.25 * min(streak, 2) if repeated_action else 0.0
        if focus == "survival" and agent_name == "Finance Co-founder":
            score += 0.2
        if focus == "crisis_recovery" and agent_name == "Finance Co-founder" and action == "fire_employee":
            score += 0.3
        if focus == "crisis_recovery" and agent_name == "Tech Co-founder" and action in {"invest_in_product", "pivot_strategy"}:
            score += 0.25
        if focus in {"product_stabilization", "reset"} and agent_name == "Tech Co-founder":
            score += 0.15
        if focus == "growth_capture" and agent_name == "Growth Co-founder":
            score += 0.2

        return score


def build_heuristic_agents() -> Tuple[TechCoFounder, GrowthCoFounder, FinanceCoFounder, CEO]:
    return TechCoFounder(), GrowthCoFounder(), FinanceCoFounder(), CEO()
