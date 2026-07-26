from __future__ import annotations

from fastapi import APIRouter

from phase5.api.routes.v1.health import router as health_router
from phase5.api.routes.v1.system import router as system_router
from phase5.api.routes.v1.knowledge_graph import router as kg_router
from phase5.api.routes.v1.reasoning import router as reasoning_router
from phase5.api.routes.v1.opportunities import router as opportunity_router
from phase5.api.routes.v1.trends import router as trend_router
from phase5.api.routes.v1.reports import router as report_router
from phase5.api.routes.v1.copilot import router as copilot_router
from phase5.api.routes.v1.search import router as search_router
from phase5.api.routes.v1.evaluation import router as evaluation_router
from phase5.api.routes.v1.exports import router as export_router
from phase5.api.routes.v1.config_mgmt import router as config_router
from phase5.api.routes.v1.monitoring import router as monitoring_router

router = APIRouter()
router.include_router(health_router, tags=["Health"])
router.include_router(system_router, tags=["System"])
router.include_router(kg_router, tags=["Knowledge Graph"])
router.include_router(reasoning_router, tags=["Reasoning"])
router.include_router(opportunity_router, tags=["Opportunities"])
router.include_router(trend_router, tags=["Trends"])
router.include_router(report_router, tags=["Reports"])
router.include_router(copilot_router, tags=["Copilot"])
router.include_router(search_router, tags=["Search"])
router.include_router(evaluation_router, tags=["Evaluation"])
router.include_router(export_router, tags=["Exports"])
router.include_router(config_router, tags=["Config"])
router.include_router(monitoring_router, tags=["Monitoring"])
