from __future__ import annotations

import time
from typing import Any, Generator

from phase4.copilot.citations.citation_system import CitationBuilder
from phase4.copilot.config import CopilotConfig
from phase4.copilot.llm.mock import MockProvider
from phase4.copilot.memory.session import SessionManager
from phase4.copilot.orchestrator.executor import ToolExecutor
from phase4.copilot.planner.planner import QueryPlanner
from phase4.copilot.response.builder import ResponseBuilder
from phase4.copilot.schema import (
    ConversationState,
    CopilotResponse,
    Intent,
    Message,
    ReasoningStep,
    ReasoningTrace,
    ResponseFormat,
    Role,
    StreamingChunk,
    ToolCall,
)


class CopilotEngine:
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()
        self._session_manager = SessionManager(self._config)
        self._planner = QueryPlanner(self._config)
        self._executor = ToolExecutor(self._config)
        self._response_builder = ResponseBuilder(self._config)
        self._citation_builder = CitationBuilder()
        self._llm = MockProvider()

    @property
    def config(self) -> CopilotConfig:
        return self._config

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        response_format: ResponseFormat | None = None,
    ) -> CopilotResponse:
        fmt = response_format or self._config.default_response_format
        info, memory, tracker, state = self._session_manager.get_or_create(session_id)
        start = time.perf_counter()

        user_msg = Message(session_id=info.session_id, role=Role.USER, content=query)
        memory.add_message(user_msg)
        state.turn_count += 1

        plan = self._planner.plan(query, state)

        if plan.intent == Intent.GREETING or plan.intent == Intent.CLARIFY:
            response = self._response_builder.build(
                query=query,
                tool_results=[],
                state=state,
                plan=plan,
                format=fmt,
            )
            memory.add_message(Message(
                session_id=info.session_id,
                role=Role.ASSISTANT,
                content=response.content,
                citations=response.citations,
            ))
            state.last_intent = plan.intent
            self._session_manager.update_state(info.session_id, state)
            return response

        tool_results = self._executor.execute_plan(plan, state)

        trace = self._build_trace(query, plan, tool_results, state, start)

        response = self._response_builder.build(
            query=query,
            tool_results=tool_results,
            state=state,
            plan=plan,
            format=fmt,
            trace=trace,
        )

        tool_calls = [
            ToolCall(
                tool_name=r.tool_name,
                result_summary=str(list(r.data.keys()))[:100] if isinstance(r.data, dict) else type(r.data).__name__,
                elapsed_ms=r.elapsed_ms,
                success=r.success,
                error=r.error,
            )
            for r in tool_results
        ]

        memory.add_message(Message(
            session_id=info.session_id,
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=tool_calls,
            citations=response.citations,
        ))

        state.last_intent = plan.intent
        state.last_tool_calls = tool_calls
        self._session_manager.update_state(info.session_id, state)

        if memory.message_count() > self._config.compression_threshold:
            memory.compress()

        return response

    def chat_stream(
        self,
        query: str,
        session_id: str | None = None,
    ) -> Generator[StreamingChunk, None, None]:
        info, memory, tracker, state = self._session_manager.get_or_create(session_id)
        start = time.perf_counter()

        user_msg = Message(session_id=info.session_id, role=Role.USER, content=query)
        memory.add_message(user_msg)
        state.turn_count += 1

        plan = self._planner.plan(query, state)

        if plan.intent == Intent.GREETING or plan.intent == Intent.CLARIFY:
            yield from self._response_builder.build_stream(query, [], state, plan)
            return

        tool_results = self._executor.execute_plan(plan, state)
        trace = self._build_trace(query, plan, tool_results, state, start)

        for chunk in self._response_builder.build_stream(query, tool_results, state, plan):
            yield chunk

        tool_calls = [
            ToolCall(
                tool_name=r.tool_name,
                result_summary=str(list(r.data.keys()))[:100] if isinstance(r.data, dict) else type(r.data).__name__,
                elapsed_ms=r.elapsed_ms,
                success=r.success,
                error=r.error,
            )
            for r in tool_results
        ]
        memory.add_message(Message(
            session_id=info.session_id,
            role=Role.ASSISTANT,
            content="[streamed]",
            tool_calls=tool_calls,
        ))
        state.last_intent = plan.intent
        self._session_manager.update_state(info.session_id, state)

    def ask(self, query: str, response_format: ResponseFormat | None = None) -> CopilotResponse:
        return self.chat(query=query, session_id=None, response_format=response_format)

    def new_session(self) -> str:
        info, _, _, _ = self._session_manager.create_session()
        return info.session_id

    def close_session(self, session_id: str) -> bool:
        return self._session_manager.close_session(session_id)

    def get_session_history(self, session_id: str) -> list[Message]:
        data = self._session_manager.get_session(session_id)
        if data is None:
            return []
        messages: list[Message] = []
        for entry in data.memory.get_full_context():
            messages.append(Message(
                session_id=session_id,
                role=entry.role,
                content=entry.content,
            ))
        return messages

    def stats(self) -> dict[str, Any]:
        return {
            "active_sessions": len(self._session_manager.list_active()),
            "total_sessions": self._session_manager.session_count(),
            "planner_confidence_threshold": self._config.planner_confidence_threshold,
            "llm_provider": self._config.llm_provider,
            "enabled_tools": self._config.enabled_tools,
            "compression_threshold": self._config.compression_threshold,
            "session_ttl_minutes": self._config.session_ttl_minutes,
        }

    def _build_trace(
        self,
        query: str,
        plan: Any,
        tool_results: Any,
        state: ConversationState,
        start: float,
    ) -> ReasoningTrace:
        steps: list[ReasoningStep] = []
        for i, node in enumerate(plan.nodes):
            r = tool_results[i] if i < len(tool_results) else None
            steps.append(ReasoningStep(
                step_type="tool",
                input=str(node.parameters),
                output=str(r.data)[:200] if r and r.success else (r.error or "") if r else "",
                tool_name=node.tool_name,
                tool_params=node.parameters,
                elapsed_ms=r.elapsed_ms if r else 0,
            ))

        trace = ReasoningTrace(
            query=query,
            intent=plan.intent,
            plan=plan,
            steps=steps,
            tool_calls=[
                ToolCall(
                    tool_name=r.tool_name,
                    result_summary=str(list(r.data.keys()))[:100] if isinstance(r.data, dict) else "",
                    elapsed_ms=r.elapsed_ms,
                    success=r.success,
                    error=r.error,
                )
                for r in tool_results
            ],
            elapsed_ms=(time.perf_counter() - start) * 1000,
            confidence=plan.confidence,
        )
        return trace
