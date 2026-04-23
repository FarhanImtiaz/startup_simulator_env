import random
from dataclasses import dataclass
from typing import Dict


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
        if quality < 0.75:
            return ActionProposal(
                action="invest_in_product",
                reasoning="Product quality is below target, so we should invest in product development.",
            )
        return ActionProposal(
            action="run_marketing_campaign",
            reasoning="Quality is acceptable; we can grow users with marketing.",
        )


class GrowthCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Growth Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        users = observation["users"]
        demand = observation["market_demand"]
        if users < 400 or demand > 0.6:
            return ActionProposal(
                action="run_marketing_campaign",
                reasoning="Market demand is strong and we should grow our user base aggressively.",
            )
        return ActionProposal(
            action="invest_in_product",
            reasoning="User growth is stable; focus on product quality to improve retention.",
        )


class FinanceCoFounder(BaseCoFounder):
    def __init__(self):
        super().__init__("Finance Co-founder")

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        money = observation["money"]
        burn_rate = observation["burn_rate"]
        if money < 30000 or burn_rate > 18000:
            return ActionProposal(
                action="fire_employee",
                reasoning="Cash reserves are low and burn is high; reduce costs by trimming the team.",
            )
        return ActionProposal(
            action="hire_employee",
            reasoning="We can afford to expand the team to support growth.",
        )


class CEO:
    def __init__(self):
        self.name = "CEO"

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        if observation["money"] < 15000:
            return proposals.get("Finance Co-founder") or random.choice(list(proposals.values()))

        if observation["product_quality"] < 0.6:
            return proposals.get("Tech Co-founder") or random.choice(list(proposals.values()))

        if observation["market_demand"] > 0.65:
            return proposals.get("Growth Co-founder") or random.choice(list(proposals.values()))

        best = max(
            proposals.values(),
            key=lambda proposal: self._score_proposal(proposal, observation),
        )
        return best

    def _score_proposal(self, proposal: ActionProposal, observation: Dict[str, object]) -> float:
        if proposal.action == "invest_in_product":
            return 1.0 + (1.0 - observation["product_quality"]) * 2.0
        if proposal.action == "run_marketing_campaign":
            return 1.0 + observation["market_demand"]
        if proposal.action == "hire_employee":
            return 1.0 if observation["money"] > 50000 else 0.6
        if proposal.action == "fire_employee":
            return 1.0 if observation["burn_rate"] > 18000 else 0.5
        if proposal.action == "pivot_strategy":
            return 0.7 if observation["market_demand"] < 0.5 else 0.4
        return 0.5
