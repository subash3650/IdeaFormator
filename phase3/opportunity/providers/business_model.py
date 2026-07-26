"""Business model providers — each evaluates fit for one OpportunityType."""

from __future__ import annotations

from phase3.opportunity.providers.base import BusinessModelProvider
from phase3.opportunity.providers.registry import register_business_model_provider
from phase3.opportunity.schema import Opportunity, OpportunityType


class _BaseBusinessModelProvider(BusinessModelProvider):
    """Shared logic: keyword-based relevance scoring."""

    _keywords: list[str] = []
    _model_type: OpportunityType = OpportunityType.SAAS

    @property
    def model_type(self) -> OpportunityType:
        return self._model_type

    def evaluate(self, opportunity: Opportunity, context: dict) -> tuple[OpportunityType, float]:
        title_lower = opportunity.title.lower()
        summary_lower = opportunity.summary.lower()
        problem_lower = opportunity.root_problem.lower()

        text = f"{title_lower} {summary_lower} {problem_lower}"
        matches = sum(1 for kw in self._keywords if kw in text)
        max_possible = len(self._keywords)
        score = min(1.0, matches / max(max_possible, 1))
        return (self._model_type, score)


@register_business_model_provider(name="saas")
class SaaSProvider(_BaseBusinessModelProvider):
    _keywords = ["subscription", "monthly", "cloud", "platform", "dashboard", "analytics", "billing"]
    _model_type = OpportunityType.SAAS
    name = "saas"


@register_business_model_provider(name="ai_agent")
class AIAgentProvider(_BaseBusinessModelProvider):
    _keywords = ["ai", "automate", "intelligent", "chatbot", "assistant", "agent", "autonomous", "ml"]
    _model_type = OpportunityType.AI_AGENT
    name = "ai_agent"


@register_business_model_provider(name="marketplace")
class MarketplaceProvider(_BaseBusinessModelProvider):
    _keywords = ["marketplace", "listing", "buyer", "seller", "platform", "exchange", "peer"]
    _model_type = OpportunityType.MARKETPLACE
    name = "marketplace"


@register_business_model_provider(name="chrome_extension")
class ChromeExtensionProvider(_BaseBusinessModelProvider):
    _keywords = ["browser", "extension", "chrome", "plugin", "addon", "bookmarklet", "tab"]
    _model_type = OpportunityType.CHROME_EXTENSION
    name = "chrome_extension"


@register_business_model_provider(name="api")
class APIProvider(_BaseBusinessModelProvider):
    _keywords = ["api", "integration", "webhook", "sdk", "rest", "graphql", "endpoint", "microservice"]
    _model_type = OpportunityType.API
    name = "api"


@register_business_model_provider(name="mobile_app")
class MobileAppProvider(_BaseBusinessModelProvider):
    _keywords = ["mobile", "app", "ios", "android", "smartphone", "tablet", "phone", "on the go"]
    _model_type = OpportunityType.MOBILE_APP
    name = "mobile_app"


@register_business_model_provider(name="b2b_platform")
class B2BPlatformProvider(_BaseBusinessModelProvider):
    _keywords = ["enterprise", "business", "b2b", "team", "org", "company", "workflow", "collaboration"]
    _model_type = OpportunityType.B2B_PLATFORM
    name = "b2b_platform"


@register_business_model_provider(name="developer_tool")
class DeveloperToolProvider(_BaseBusinessModelProvider):
    _keywords = ["developer", "code", "deploy", "debug", "ci/cd", "cli", "framework", "library"]
    _model_type = OpportunityType.DEVELOPER_TOOL
    name = "developer_tool"


@register_business_model_provider(name="consumer_product")
class ConsumerProductProvider(_BaseBusinessModelProvider):
    _keywords = ["consumer", "user", "personal", "home", "family", "individual", "lifestyle"]
    _model_type = OpportunityType.CONSUMER_PRODUCT
    name = "consumer_product"
