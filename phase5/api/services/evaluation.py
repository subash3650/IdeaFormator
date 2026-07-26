from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class EvaluationService(BaseEngineService):
    async def _get_evaluator(self):
        from phase4.copilot.evaluation.evaluator import CopilotEvaluator
        from phase4.copilot.config import CopilotConfig
        config = CopilotConfig(knowledge_dir=str(self._knowledge_dir))
        return CopilotEvaluator(config)

    async def run_benchmark(self, benchmark_path: str) -> dict[str, Any]:
        evaluator = await self._get_evaluator()
        return await self._run_in_thread(evaluator.evaluate_benchmark, benchmark_path)

    async def evaluate_intents(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
        evaluator = await self._get_evaluator()
        from phase4.copilot.schema import Intent
        cases = [(t["query"], Intent(t["expected_intent"])) for t in test_cases]
        return await self._run_in_thread(evaluator.evaluate_intent_classification, cases)

    async def stats(self) -> dict[str, Any]:
        return {"available_benchmarks": ["phase4/copilot/benchmarks/"], "status": "ready"}
