from phase3.presentation.templates.base import BaseTemplate
from phase3.presentation.templates.registry import (
    available_templates,
    create_template,
    get_template_class,
    register_template,
)

from phase3.presentation.templates.business import BusinessTemplate
from phase3.presentation.templates.executive import ExecutiveTemplate
from phase3.presentation.templates.investor import InvestorTemplate
from phase3.presentation.templates.founder import FounderTemplate
from phase3.presentation.templates.technology import TechnologyTemplate
from phase3.presentation.templates.market import MarketTemplate

__all__ = [
    "BaseTemplate",
    "register_template",
    "get_template_class",
    "create_template",
    "available_templates",
    "BusinessTemplate",
    "ExecutiveTemplate",
    "InvestorTemplate",
    "FounderTemplate",
    "TechnologyTemplate",
    "MarketTemplate",
]
