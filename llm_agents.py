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
                "You are the Tech Co-founder of a startup. "
                "Prioritize product quality, stability, and long-term retention."
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
                "You are the Growth Co-founder of a startup. "
                "Prioritize user growth, market capture, and momentum."
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
                "You are the Finance Co-founder of a startup. "
                "Prioritize cash preservation, runway, and operational sustainability."
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
            "Respond with exactly one line in the form: Action: <action_name>"
        )
        return PromptArtifacts(
            system_prompt=(
                "You are the CEO of a startup. "
                "Choose the single best action for long-term company success under uncertainty."
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
        f"- Ad Performance: {observation['ad_performance']}\n"
        f"- Runway Hint: {observation['runway_hint']}\n"
        f"- Recent Events: {observation['recent_events']}\n"
        f"- Recent Actions: {observation['recent_actions']}"
    )
