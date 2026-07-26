from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _compute_id(prefix: str, *parts: str) -> str:
    raw = prefix + "-" + "-".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Intent(str, Enum):
    QUERY_KG = "query_kg"
    QUERY_REASONING = "query_reasoning"
    QUERY_OPPORTUNITY = "query_opportunity"
    QUERY_TREND = "query_trend"
    QUERY_PRESENTATION = "query_presentation"
    COMPARE = "compare"
    EXPLAIN = "explain"
    EVIDENCE = "evidence"
    SEARCH = "search"
    BRIEFING = "briefing"
    STATISTICS = "statistics"
    CLARIFY = "clarify"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class CitationSource(str, Enum):
    KNOWLEDGE_GRAPH = "knowledge_graph"
    REASONING = "reasoning"
    OPPORTUNITY = "opportunity"
    TREND = "trend"
    PRESENTATION = "presentation"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    STREAMING = "streaming"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    CLOSED = "closed"


class MemoryLevel(str, Enum):
    SHORT_TERM = "short_term"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    PINNED = "pinned"


class PermissionType(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class ToolPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComparisonEntityType(str, Enum):
    COMPANY = "company"
    PRODUCT = "product"
    OPPORTUNITY = "opportunity"
    TREND = "trend"


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    entry_id: str = ""
    role: Role
    content: str
    level: MemoryLevel = MemoryLevel.SHORT_TERM
    timestamp: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "entry_id" not in data or not data["entry_id"]:
            data["entry_id"] = _compute_id("mem", data.get("role", ""), str(time.time_ns()))
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = _now_iso()
        super().__init__(**data)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_call_id: str = ""
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    elapsed_ms: float = 0.0
    success: bool = True
    error: str | None = None

    def __init__(self, **data: Any) -> None:
        if "tool_call_id" not in data or not data["tool_call_id"]:
            data["tool_call_id"] = _compute_id("tc", data.get("tool_name", ""), str(time.time_ns()))
        super().__init__(**data)


class ToolMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    description: str
    supported_intents: list[Intent]
    priority: ToolPriority = ToolPriority.MEDIUM
    estimated_cost: float = 1.0
    estimated_latency_ms: float = 100.0
    dependencies: list[str] = Field(default_factory=list)
    permissions: PermissionType = PermissionType.READ_ONLY
    supports_streaming: bool = False
    cacheable: bool = True
    max_concurrent: int = 1


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    citation_id: str = ""
    source_module: CitationSource
    source_id: str
    source_title: str = ""
    confidence: float = 0.0
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "citation_id" not in data or not data["citation_id"]:
            data["citation_id"] = _compute_id("cit", data.get("source_module", ""), data.get("source_id", ""))
        super().__init__(**data)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tool_name: str
    success: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None


class PlanNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: str = ""
    step_index: int
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    description: str = ""

    def __init__(self, **data: Any) -> None:
        if "node_id" not in data or not data["node_id"]:
            data["node_id"] = _compute_id("plan", str(data.get("step_index", 0)), data.get("tool_name", ""))
        super().__init__(**data)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    plan_id: str = ""
    intent: Intent
    nodes: list[PlanNode] = Field(default_factory=list)
    context_hints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    def __init__(self, **data: Any) -> None:
        if "plan_id" not in data or not data["plan_id"]:
            data["plan_id"] = _compute_id("exec", str(data.get("intent", "")), str(time.time_ns()))
        super().__init__(**data)


class ReasoningStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    step_type: str
    input: str = ""
    output: str = ""
    tool_name: str | None = None
    tool_params: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class ReasoningTrace(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    trace_id: str = ""
    query: str
    intent: Intent
    plan: ExecutionPlan | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: list[ReasoningStep] = Field(default_factory=list)
    response_id: str = ""
    elapsed_ms: float = 0.0
    confidence: float = 1.0

    def __init__(self, **data: Any) -> None:
        if "trace_id" not in data or not data["trace_id"]:
            data["trace_id"] = _compute_id("trace", str(data.get("intent", "")), str(time.time_ns()))
        super().__init__(**data)


class CopilotResponse(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    response_id: str = ""
    session_id: str = ""
    content: str = ""
    format: ResponseFormat = ResponseFormat.MARKDOWN
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    trace: ReasoningTrace | None = None
    confidence: float = 1.0
    suggested_followups: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "response_id" not in data or not data["response_id"]:
            data["response_id"] = _compute_id("resp", str(data.get("session_id", "")), str(time.time_ns()))
        super().__init__(**data)


class ConversationState(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    session_id: str
    turn_count: int = 0
    last_intent: Intent | None = None
    last_tool_calls: list[ToolCall] = Field(default_factory=list)
    mentioned_entities: list[str] = Field(default_factory=list)
    mentioned_opportunities: list[str] = Field(default_factory=list)
    mentioned_trends: list[str] = Field(default_factory=list)
    topic_stack: list[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    session_id: str
    created_at: str = ""
    last_active_at: str = ""
    message_count: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    topic_summary: str = ""

    def __init__(self, **data: Any) -> None:
        now = _now_iso()
        if "created_at" not in data or not data["created_at"]:
            data["created_at"] = now
        if "last_active_at" not in data or not data["last_active_at"]:
            data["last_active_at"] = now
        super().__init__(**data)


class LLMProviderResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    content: str
    provider: str = "mock"
    model: str = "mock"
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: float = 0.0


class BenchmarkQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    category: str
    query: str
    expected_intent: Intent
    expected_tools: list[str] = Field(default_factory=list)
    min_citations: int = 0
    session_id: str | None = None


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    query_id: str
    category: str
    query: str
    intent_matched: bool = False
    tools_used: list[str] = Field(default_factory=list)
    expected_tools_matched: bool = False
    citation_count: int = 0
    min_citations_met: bool = False
    elapsed_ms: float = 0.0
    success: bool = False
    error: str | None = None


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    message_id: str = ""
    session_id: str = ""
    role: Role
    content: str = ""
    timestamp: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        if "message_id" not in data or not data["message_id"]:
            data["message_id"] = _compute_id("msg", str(data.get("session_id", "")), str(data.get("role", "")))
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = _now_iso()
        super().__init__(**data)


class ConversationMemoryState(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")
    short_term: list[MemoryEntry] = Field(default_factory=list)
    conversation: list[MemoryEntry] = Field(default_factory=list)
    long_term: list[MemoryEntry] = Field(default_factory=list)
    pinned: list[MemoryEntry] = Field(default_factory=list)
    compressed_summary: str = ""
    needs_compression: bool = False

    def total_entries(self) -> int:
        return len(self.short_term) + len(self.conversation) + len(self.long_term) + len(self.pinned)


class StreamingChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_type: str = "token"
    data: str = ""
    index: int = 0
    final: bool = False
