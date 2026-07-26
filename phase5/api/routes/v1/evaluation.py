from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from phase5.api.dependencies.auth import require_admin, require_read
from phase5.api.dependencies.engine import get_knowledge_dir
from phase5.api.responses.envelope import build_success, build_error
from phase5.api.services.evaluation import EvaluationService

router = APIRouter(prefix="/evaluation")


@router.post("/benchmark", summary="Run a benchmark evaluation")
async def run_benchmark(
    request: Request,
    benchmark_path: str = Query(default="", description="Path to benchmark JSON file"),
    user=Depends(require_admin),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    if not benchmark_path:
        return JSONResponse(status_code=422, content=build_error("VALIDATION", "benchmark_path is required").model_dump(mode="json"))
    service = EvaluationService(knowledge_dir)
    result = await service.run_benchmark(benchmark_path)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.post("/intents", summary="Evaluate intent classification")
async def evaluate_intents(
    request: Request,
    test_cases: list[dict] = [],
    user=Depends(require_admin),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    if not test_cases:
        return JSONResponse(status_code=422, content=build_error("VALIDATION", "test_cases is required").model_dump(mode="json"))
    service = EvaluationService(knowledge_dir)
    result = await service.evaluate_intents(test_cases)
    return JSONResponse(content=build_success(result).model_dump(mode="json"))


@router.get("/stats", summary="Evaluation statistics")
async def evaluation_stats(
    request: Request,
    user=Depends(require_read),
    knowledge_dir: Path = Depends(get_knowledge_dir),
) -> JSONResponse:
    service = EvaluationService(knowledge_dir)
    stats = await service.stats()
    return JSONResponse(content=build_success(stats).model_dump(mode="json"))
