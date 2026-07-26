"""Rule registry with metadata (name, version, priority, dependencies)."""

from __future__ import annotations

from phase2.reasoning.rules.base import ReasoningRule, RuleMetadata

_rules: dict[str, tuple[type[ReasoningRule], RuleMetadata]] = {}


def register_rule(
    name: str,
    version: str = "1.0",
    priority: int = 50,
    dependencies: list[str] | None = None,
    description: str = "",
    author: str = "system",
):
    def wrapper(cls: type[ReasoningRule]) -> type[ReasoningRule]:
        _rules[name] = (
            cls,
            RuleMetadata(
                name=name,
                version=version,
                priority=priority,
                dependencies=dependencies or [],
                description=description or cls.__doc__ or "",
                author=author,
            ),
        )
        return cls
    return wrapper


def get_rule_class(name: str) -> type[ReasoningRule]:
    if name not in _rules:
        available = ", ".join(sorted(_rules))
        msg = f"Unknown rule {name!r}. Available: {available}"
        raise KeyError(msg)
    return _rules[name][0]


def create_rule(name: str, **kwargs) -> ReasoningRule:
    cls = get_rule_class(name)
    return cls(**kwargs)


def get_rule_metadata(name: str) -> RuleMetadata:
    if name not in _rules:
        available = ", ".join(sorted(_rules))
        msg = f"Unknown rule {name!r}. Available: {available}"
        raise KeyError(msg)
    return _rules[name][1]


def available_rules() -> list[str]:
    return sorted(_rules.keys())


def get_rule_priorities() -> dict[str, int]:
    return {name: meta.priority for name, (_, meta) in _rules.items()}


def get_rules_sorted_by_priority() -> list[tuple[str, type[ReasoningRule], RuleMetadata]]:
    items = [(name, cls, meta) for name, (cls, meta) in _rules.items()]
    items.sort(key=lambda x: x[2].priority, reverse=True)
    return items
