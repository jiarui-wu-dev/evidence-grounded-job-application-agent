"""Local FastAPI entry point for the application workflow."""

from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.api.evidence_schema import EvidenceResponse
from src.api.review_schema import EvidenceReviewRequest, RequirementReviewRequest, ReviewResponse
from src.services.applications import (
    ApplicationService, ReviewValidationFailed, RunConflict, RunNotFound, configured_run_database,
)


FinalStatus = Literal["grounded", "needs_human_review"]
JobDescription = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30000)]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jd_text: JobDescription


class ApplicationRequest(JobRequest):
    max_revision_rounds: int = Field(default=2, strict=True, ge=0, le=2)


class RunResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    final_status: FinalStatus | None
    error_code: str | None
    created_at: str
    updated_at: str


class TraceEvent(BaseModel):
    created_at: str | None
    next_nodes: list[str]
    revision_round: int | None
    pending_revalidation_claim_ids: list[str] | None
    validation_counts: dict[str, int]
    final_status: FinalStatus | None


class TraceResponse(BaseModel):
    run_id: str
    events: list[TraceEvent]


def create_app(service: ApplicationService | None = None) -> FastAPI:
    if service is None:
        service = ApplicationService(configured_run_database())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.initialize()
        yield

    app = FastAPI(title="Evidence-Grounded Job Application Agent", lifespan=lifespan)

    @app.exception_handler(RunNotFound)
    async def missing_run(_: Request, __: RunNotFound):
        return JSONResponse(status_code=404, content={"detail": "Run not found."})

    @app.exception_handler(RunConflict)
    async def run_conflict(_: Request, error: RunConflict):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def invalid_input(_: Request, __: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @app.exception_handler(ReviewValidationFailed)
    async def review_validation_failed(_: Request, __: ReviewValidationFailed):
        return JSONResponse(status_code=502, content={"detail": "Review validation failed. Saved decisions are intact."})

    @app.post("/applications", status_code=status.HTTP_202_ACCEPTED, response_model=RunResponse)
    def create_application(payload: ApplicationRequest, tasks: BackgroundTasks):
        run = service.create_run(payload.jd_text, payload.max_revision_rounds)
        tasks.add_task(service.execute_run, run["run_id"])
        return run

    @app.post("/analyze-job")
    def analyze_job(payload: JobRequest) -> dict[str, Any]:
        try:
            return service.analyze_job(payload.jd_text)
        except Exception:
            raise HTTPException(status_code=502, detail="Job analysis failed.") from None

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str):
        return service.get_run(run_id)

    @app.get("/runs/{run_id}/trace", response_model=TraceResponse)
    def get_trace(run_id: str):
        return {"run_id": run_id, "events": service.get_trace(run_id)}

    @app.get("/runs/{run_id}/cv")
    def get_cv(run_id: str) -> dict[str, Any]:
        return service.get_cv(run_id)

    @app.get("/runs/{run_id}/evidence", response_model=EvidenceResponse)
    def get_evidence(run_id: str) -> dict[str, Any]:
        return service.get_evidence(run_id)

    @app.get("/runs/{run_id}/review", response_model=ReviewResponse)
    def get_review(run_id: str):
        return service.get_review_queue(run_id)

    @app.post("/runs/{run_id}/review/evidence/{evidence_id}", response_model=ReviewResponse)
    def decide_evidence(run_id: str, evidence_id: str, payload: EvidenceReviewRequest):
        if payload.decision == "confirm":
            return service.confirm_evidence(run_id, evidence_id, payload.note)
        return service.reject_evidence(run_id, evidence_id, payload.note)

    @app.post("/runs/{run_id}/review/requirements/{requirement_id}", response_model=ReviewResponse)
    def decide_requirement(run_id: str, requirement_id: str, payload: RequirementReviewRequest):
        return service.resolve_requirement(run_id, requirement_id, payload.decision, payload.note)

    @app.post("/runs/{run_id}/review/complete", response_model=ReviewResponse)
    def complete_review(run_id: str):
        return service.complete_review(run_id)

    @app.post("/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED, response_model=RunResponse)
    def resume(run_id: str, tasks: BackgroundTasks):
        run = service.queue_resume(run_id)
        tasks.add_task(service.execute_run, run_id)
        return run

    return app


app = create_app()
