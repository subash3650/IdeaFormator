from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.schema import ConversationState, ExecutionPlan, PlanNode, ReasoningStep, ToolCall, ToolResult
from phase4.copilot.tools.registry import create_tool


class ToolExecutor:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._max_workers = 4

    def execute_plan(
        self,
        plan: ExecutionPlan,
        state: ConversationState,
    ) -> list[ToolResult]:
        if not plan.nodes:
            return []

        results: dict[int, ToolResult] = {}
        trace_steps: list[ReasoningStep] = []

        executed: set[int] = set()
        remaining = set(range(len(plan.nodes)))

        while remaining:
            ready = [
                i for i in remaining
                if all(dep in executed for dep in plan.nodes[i].depends_on)
            ]

            if not ready:
                break

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                future_to_index = {
                    pool.submit(self._execute_node, plan.nodes[i], state): i
                    for i in ready
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result, step = future.result()
                        results[idx] = result
                        trace_steps.append(step)
                        executed.add(idx)
                        remaining.remove(idx)
                    except Exception as e:
                        results[idx] = ToolResult(
                            tool_name=plan.nodes[idx].tool_name,
                            success=False,
                            data={},
                            error=str(e),
                        )
                        executed.add(idx)
                        remaining.remove(idx)

        return [results[i] for i in sorted(results)]

    def execute_node_sync(self, node: PlanNode, state: ConversationState) -> tuple[ToolResult, ReasoningStep]:
        return self._execute_node(node, state)

    def _execute_node(self, node: PlanNode, state: ConversationState) -> tuple[ToolResult, ReasoningStep]:
        start = time.perf_counter()
        try:
            tool = create_tool(node.tool_name, config=self._config)
            result = tool.execute(node.parameters)
        except KeyError:
            result = ToolResult(
                tool_name=node.tool_name,
                success=False,
                data={},
                error=f"Unknown tool: {node.tool_name}",
            )
        except Exception as e:
            result = ToolResult(
                tool_name=node.tool_name,
                success=False,
                data={},
                error=str(e),
            )

        elapsed = (time.perf_counter() - start) * 1000
        step = ReasoningStep(
            step_type="tool_call",
            input=str(node.parameters),
            output=str(result.data)[:200] if result.success else result.error or "",
            tool_name=node.tool_name,
            tool_params=node.parameters,
            citations=[c.citation_id for c in result.citations],
            elapsed_ms=round(elapsed, 1),
        )

        return result, step
