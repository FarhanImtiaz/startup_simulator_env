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


class TechCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Tech Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        quality = observation["product_quality"]
        recent_events = observation["recent_events"]

        if "tech_failure" in recent_events or quality < 0.62:
            return ActionProposal(
                action="invest_in_product",
                reasoning="Quality risk is showing up in product signals, so we should stabilize the product first.",
            )

        if observation["recent_user_growth"] < 0 and quality < 0.75:
            return ActionProposal(
                action="invest_in_product",
                reasoning="Growth is soft and quality is not strong enough to defend retention yet.",
            )

        return ActionProposal(
            action="hire_employee",
            reasoning="The product is in decent shape, so adding capacity should unlock future improvements.",
        )


class GrowthCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Growth Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        ad_performance = observation["ad_performance"]
        growth = observation["recent_user_growth"]
        recent_events = observation["recent_events"]

        if "viral_growth" in recent_events:
            return ActionProposal(
                action="run_marketing_campaign",
                reasoning="Momentum is already hot, so we should press the advantage while attention is high.",
            )

        if ad_performance != "poor" and (growth < 40 or observation["users"] < 450):
            return ActionProposal(
                action="run_marketing_campaign",
                reasoning="Demand signals still look usable, and we need more users to grow.",
            )

        return ActionProposal(
            action="pivot_strategy",
            reasoning="Current growth signals look weak, so we should reposition before spending more on acquisition.",
        )


class FinanceCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Finance Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        money = observation["money"]
        burn_rate = observation["burn_rate"]
        runway_hint = observation["runway_hint"]

        if money < 25000 or runway_hint < 2.2:
            return ActionProposal(
                action="fire_employee",
                reasoning="Cash runway looks tight, so we should cut burn immediately.",
            )

        if burn_rate > 18000 and observation["recent_user_growth"] < 0:
            return ActionProposal(
                action="do_nothing",
                reasoning="The company is spending heavily without healthy growth, so we should pause and protect cash.",
            )

        return ActionProposal(
            action="hire_employee",
            reasoning="The balance sheet can still support measured team expansion.",
        )


class CEO:
    def __init__(self):
        self.name = "CEO"

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        if observation["money"] < 12000 or observation["runway_hint"] < 1.8:
            return proposals.get("Finance Co-founder") or random.choice(list(proposals.values()))

        if "tech_failure" in observation["recent_events"] or observation["product_quality"] < 0.55:
            return proposals.get("Tech Co-founder") or random.choice(list(proposals.values()))

        if "viral_growth" in observation["recent_events"] or observation["ad_performance"] == "good":
            return proposals.get("Growth Co-founder") or random.choice(list(proposals.values()))

        best = max(
            proposals.values(),
            key=lambda proposal: self._score_proposal(proposal, observation),
        )
        return best

    def _score_proposal(self, proposal: ActionProposal, observation: Dict[str, object]) -> float:
        if proposal.action == "invest_in_product":
            return 1.2 + (1.0 - observation["product_quality"]) * 1.8
        if proposal.action == "run_marketing_campaign":
            score = 1.0
            if observation["ad_performance"] == "good":
                score += 0.7
            if observation["recent_user_growth"] < 0:
                score -= 0.3
            return score
        if proposal.action == "hire_employee":
            return 0.9 if observation["runway_hint"] > 4 else 0.5
        if proposal.action == "fire_employee":
            return 1.1 if observation["runway_hint"] < 2.5 else 0.45
        if proposal.action == "pivot_strategy":
            return 1.0 if observation["recent_user_growth"] < 0 else 0.6
        if proposal.action == "do_nothing":
            return 0.8 if observation["burn_rate"] > 18000 else 0.3
        return 0.2


def build_heuristic_agents() -> Tuple[TechCoFounder, GrowthCoFounder, FinanceCoFounder, CEO]:
    return TechCoFounder(), GrowthCoFounder(), FinanceCoFounder(), CEO()
