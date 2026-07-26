from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from phase4.copilot.config import CopilotConfig
from phase4.copilot.engine import CopilotEngine
from phase4.copilot.planner.intent import IntentClassifier
from phase4.copilot.schema import BenchmarkQuery, BenchmarkResult, Intent


class CopilotEvaluator:
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()
        self._engine = CopilotEngine(self._config)
        self._classifier = IntentClassifier()

    def evaluate_benchmark(self, benchmark_path: str | Path) -> dict[str, Any]:
        path = Path(benchmark_path)
        if not path.exists():
            return {"error": f"Benchmark file not found: {benchmark_path}"}

        with open(str(path), encoding="utf-8") as f:
            raw = json.load(f)

        queries = [BenchmarkQuery(**q) for q in raw.get("queries", [])]
        results: list[BenchmarkResult] = []
        total_start = time.perf_counter()

        for q in queries:
            result = self._evaluate_single(q)
            results.append(result)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        passed = sum(1 for r in results if r.success)
        intent_matched = sum(1 for r in results if r.intent_matched)
        tools_matched = sum(1 for r in results if r.expected_tools_matched)
        citations_met = sum(1 for r in results if r.min_citations_met)

        return {
            "total_queries": len(queries),
            "passed": passed,
            "failed": len(queries) - passed,
            "pass_rate": round(passed / len(queries) * 100, 1) if queries else 0,
            "intent_accuracy": round(intent_matched / len(queries) * 100, 1) if queries else 0,
            "tool_accuracy": round(tools_matched / len(queries) * 100, 1) if queries else 0,
            "citation_rate": round(citations_met / len(queries) * 100, 1) if queries else 0,
            "avg_elapsed_ms": round(total_elapsed / len(queries), 1) if queries else 0,
            "results": [r.model_dump() for r in results],
        }

    def evaluate_intent_classification(self, test_cases: list[tuple[str, Intent]]) -> dict[str, Any]:
        correct = 0
        results: list[dict[str, Any]] = []
        for query, expected in test_cases:
            actual, confidence = self._classifier.classify(query)
            matched = actual == expected
            if matched:
                correct += 1
            results.append({
                "query": query,
                "expected": expected.value,
                "actual": actual.value,
                "confidence": round(confidence, 2),
                "matched": matched,
            })
        return {
            "total": len(test_cases),
            "correct": correct,
            "accuracy": round(correct / len(test_cases) * 100, 1) if test_cases else 0,
            "results": results,
        }

    def evaluate_tool_execution(self, queries: list[str]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for query in queries:
            try:
                start = time.perf_counter()
                response = self._engine.ask(query)
                elapsed = (time.perf_counter() - start) * 1000
                results.append({
                    "query": query,
                    "success": True,
                    "tool_count": len(response.tool_calls),
                    "citation_count": len(response.citations),
                    "content_length": len(response.content),
                    "elapsed_ms": round(elapsed, 1),
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                })
        return {
            "total": len(queries),
            "successful": sum(1 for r in results if r.get("success")),
            "results": results,
        }

    def _evaluate_single(self, q: BenchmarkQuery) -> BenchmarkResult:
        start = time.perf_counter()
        try:
            actual_intent, _ = self._classifier.classify(q.query)
            intent_matched = actual_intent == q.expected_intent

            response = self._engine.ask(q.query)
            elapsed = (time.perf_counter() - start) * 1000

            tools_used = [tc.tool_name for tc in response.tool_calls]
            expected_tools_matched = all(t in tools_used for t in q.expected_tools)

            citation_count = len(response.citations)
            min_citations_met = citation_count >= q.min_citations

            success = True

            return BenchmarkResult(
                query_id=q.id,
                category=q.category,
                query=q.query,
                intent_matched=intent_matched,
                tools_used=tools_used,
                expected_tools_matched=expected_tools_matched,
                citation_count=citation_count,
                min_citations_met=min_citations_met,
                elapsed_ms=round(elapsed, 1),
                success=success,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                query_id=q.id,
                category=q.category,
                query=q.query,
                success=False,
                error=str(e),
                elapsed_ms=round(elapsed, 1),
            )
