import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from agents import ActionProposal, CEO, FinanceCoFounder, GrowthCoFounder, TechCoFounder
from environment import StartupEnvironment


ActionGenerator = Callable[[str], str]


@dataclass
class PromptArtifacts:
    system_prompt: str
    user_prompt: str


class PromptedAgentMixin:
    def __init__(
        self,
        name: str,
        fallback_agent,
        allowed_actions,
        generator: Optional[ActionGenerator] = None,
    ):
        self.name = name
        self.fallback_agent = fallback_agent
        self.allowed_actions = tuple(allowed_actions)
        self.generator = generator
        self.last_prompt: Optional[PromptArtifacts] = None
        self.last_raw_response: Optional[str] = None

    def propose(self, observation: Dict[str, object]) -> ActionProposal:
        fallback = self.fallback_agent.propose(observation)
        prompt = self.build_prompt(observation, fallback)
        self.last_prompt = prompt

        if self.generator is None:
            self.last_raw_response = None
            return fallback

        raw_response = self.generator(self._compose_prompt(prompt))
        self.last_raw_response = raw_response
        action = parse_action(raw_response, self.allowed_actions)
        if action is None:
            return fallback
        if (
            (observation.get("crisis_level") == "crisis" or observation.get("runway_hint", 999) < 2)
            and action in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS
        ):
            if fallback.action not in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS:
                return fallback
            return ActionProposal(
                action="do_nothing",
                reasoning=f"Prompt safety fallback: {action} is disallowed in crisis.",
            )

        return ActionProposal(action=action, reasoning=f"LLM-selected action from prompt scaffold: {action}")

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        raise NotImplementedError

    @staticmethod
    def _compose_prompt(prompt: PromptArtifacts) -> str:
        return f"{prompt.system_prompt}\n\n{prompt.user_prompt}"


class PromptedTechCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Tech Co-founder",
            fallback_agent=TechCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Tech Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize product quality, stability, and long-term retention, and avoid mindlessly repeating the same action unless the evidence is still strong. "
                "If the company is in crisis, prefer decisive recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedGrowthCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Growth Co-founder",
            fallback_agent=GrowthCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Growth Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize user growth, market capture, and momentum, and avoid repeating stale growth plays when recent evidence weakens. "
                "If the company is in crisis, prefer decisive recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedFinanceCoFounder(PromptedAgentMixin):
    def __init__(self, generator: Optional[ActionGenerator] = None):
        super().__init__(
            name="Finance Co-founder",
            fallback_agent=FinanceCoFounder(),
            allowed_actions=StartupEnvironment.ACTIONS,
            generator=generator,
        )

    def build_prompt(self, observation: Dict[str, object], fallback: ActionProposal) -> PromptArtifacts:
        return PromptArtifacts(
            system_prompt=(
                "You are the Finance Co-founder of a startup operating under uncertainty. "
                "Use recent trends, the last three growth signals, runway hints, recent actions, and recent events to infer hidden conditions. "
                "Prioritize cash preservation, runway, and operational sustainability, but avoid getting stuck in repetitive cost-cutting when the business may need recovery. "
                "If the company is in crisis, prefer decisive survival-and-recovery moves over passive waiting."
            ),
            user_prompt=_format_role_prompt(observation, self.allowed_actions, fallback.action),
        )


class PromptedCEO:
    def __init__(
        self,
        generator: Optional[ActionGenerator] = None,
        fallback_agent: Optional[CEO] = None,
    ):
        self.name = "CEO"
        self.generator = generator
        self.fallback_agent = fallback_agent or CEO()
        self.allowed_actions = tuple(StartupEnvironment.ACTIONS)
        self.last_prompt: Optional[PromptArtifacts] = None
        self.last_raw_response: Optional[str] = None

    def choose_action(self, proposals: Dict[str, ActionProposal], observation: Dict[str, object]) -> ActionProposal:
        fallback = self.fallback_agent.choose_action(proposals, observation)
        prompt = self.build_prompt(proposals, observation, fallback)
        self.last_prompt = prompt

        if self.generator is None:
            self.last_raw_response = None
            return fallback

        raw_response = self.generator(f"{prompt.system_prompt}\n\n{prompt.user_prompt}")
        self.last_raw_response = raw_response
        action = parse_action(raw_response, self.allowed_actions)
        if action is None:
            return fallback
        if (
            (observation.get("crisis_level") == "crisis" or observation.get("runway_hint", 999) < 2)
            and action in StartupEnvironment.CRISIS_DISALLOWED_ACTIONS
        ):
            return ActionProposal(
                action="fire_employee",
                reasoning=f"LLM safety override: {action} is disallowed in crisis, so selected fire_employee.",
            )
        return ActionProposal(action=action, reasoning=f"LLM-selected final decision: {action}")

    def build_prompt(
        self,
        proposals: Dict[str, ActionProposal],
        observation: Dict[str, object],
        fallback: ActionProposal,
    ) -> PromptArtifacts:
        proposal_lines = [
            f"- {agent_name}: {proposal.action} | {proposal.reasoning}"
            for agent_name, proposal in proposals.items()
        ]
        user_prompt = (
            f"{_format_observation(observation)}\n\n"
            "Co-founder proposals:\n"
            f"{chr(10).join(proposal_lines)}\n\n"
            f"Allowed actions: {', '.join(self.allowed_actions)}\n"
            f"Fallback action if uncertain: {fallback.action}\n"
            "Strict CEO priority order: Survival, then Recovery, then Optimization.\n"
            "Survival mode: if runway is below 2, reduce burn immediately with fire_employee if possible; never choose run_marketing_campaign or hire_employee.\n"
            "Recovery mode: if growth has been negative for 3 consecutive turns, do not wait; choose invest_in_product unless product was recently invested in, otherwise pivot_strategy.\n"
            "Anti-freeze: never choose the same action more than 2 times in a row; if last action was do_nothing and growth is negative, do_nothing is forbidden.\n"
            "Decline response: if last_3_growth is mostly negative or users are consistently decreasing, intervene with invest_in_product by default or pivot_strategy if decline continues.\n"
            "Runway rules: above 6 allows product/growth and rare hiring; 3 to 6 avoids hiring and prefers product; below 3 prioritizes cash preservation.\n"
            "Growth capture: if growth is strongly positive and trend is improving, choose run_marketing_campaign unless a higher-priority survival or recovery rule applies.\n"
            "Default: choose invest_in_product; do not default to do_nothing.\n"
            "Output reasoning should cover mode, trigger, selected action, and rejected options, then include Action: <action_name>.\n"
            "Respond with exactly one line in the form: Action: <action_name>"
        )
        return PromptArtifacts(
            system_prompt=(
                "You are the CEO of a startup operating under uncertainty. "
                "Choose the single best action for long-term company success by weighing runway hints, growth trend, recent events, crisis status, and the risk of repeating an action loop. "
                "If the company is in crisis, avoid passive waiting unless there is truly no viable alternative. "
                "Use the co-founder proposals as inputs, not commands."
            ),
            user_prompt=user_prompt,
        )


def build_prompted_agents(
    generator: Optional[ActionGenerator] = None,
) -> Tuple[PromptedTechCoFounder, PromptedGrowthCoFounder, PromptedFinanceCoFounder, PromptedCEO]:
    return (
        PromptedTechCoFounder(generator=generator),
        PromptedGrowthCoFounder(generator=generator),
        PromptedFinanceCoFounder(generator=generator),
        PromptedCEO(generator=generator),
    )


def parse_action(text: Optional[str], allowed_actions) -> Optional[str]:
    if not text:
        return None

    lowered = text.strip().lower()
    allowed = {action.lower(): action for action in allowed_actions}

    match = re.search(r"action\s*:\s*([a-z_]+)", lowered)
    if match and match.group(1) in allowed:
        return allowed[match.group(1)]

    for token in re.findall(r"[a-z_]+", lowered):
        if token in allowed:
            return allowed[token]
    return None


def _format_role_prompt(observation: Dict[str, object], allowed_actions, fallback_action: str) -> str:
    return (
        f"{_format_observation(observation)}\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}\n"
        f"Fallback action if uncertain: {fallback_action}\n"
        "Reason silently about trend direction, uncertainty, and whether repeating the last action is justified.\n"
        "If crisis mode is active, favor a decisive recovery action instead of do_nothing unless every active option is clearly worse.\n"
        "If runway is below 2, do not choose run_marketing_campaign or hire_employee.\n"
        "Do not hire unless runway is above 6, all last 3 growth values are positive, and recent_events does not include viral_growth.\n"
        "Respond with exactly one line in the form: Action: <action_name>"
    )


def _format_observation(observation: Dict[str, object]) -> str:
    return (
        "Observed startup state:\n"
        f"- Day: {observation['day']}\n"
        f"- Money: {observation['money']}\n"
        f"- Users: {observation['users']}\n"
        f"- Product Quality: {observation['product_quality']}\n"
        f"- Team Size: {observation['team_size']}\n"
        f"- Burn Rate: {observation['burn_rate']}\n"
        f"- Recent User Growth Signal: {observation['recent_user_growth']}\n"
        f"- Last 3 Growth Signals: {observation['last_3_growth']}\n"
        f"- Trend Direction: {observation['trend_direction']}\n"
        f"- Ad Performance: {observation['ad_performance']}\n"
        f"- Runway Hint: {observation['runway_hint']}\n"
        f"- Crisis Mode: {observation['is_crisis']}\n"
        f"- Crisis Level: {observation['crisis_level']}\n"
        f"- Crisis Reason: {observation['crisis_reason']}\n"
        f"- Recent Events: {observation['recent_events']}\n"
        f"- Recent Actions: {observation['recent_actions']}\n"
        f"- Last Action: {observation['last_action']}\n"
        f"- Consecutive Action Streak: {observation['consecutive_action_streak']}"
    )
