"""Explicit requests and audit response for human review."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceDecision = Literal["confirm", "reject"]
RequirementDecision = Literal["keep_gap", "ignore_for_application"]
Decision = Literal["confirm", "reject", "keep_gap", "ignore_for_application"]
ReviewStatus = Literal["grounded", "needs_human_review"]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(default=None, max_length=2000)


class EvidenceReviewRequest(ReviewRequest):
    decision: EvidenceDecision


class RequirementReviewRequest(ReviewRequest):
    decision: RequirementDecision


class ReviewDecision(BaseModel):
    review_id: str
    run_id: str
    target_type: Literal["evidence", "requirement"]
    target_id: str
    decision: Decision
    previous_status: str
    new_status: str
    timestamp: str
    note: str | None


class ReviewItem(BaseModel):
    target_type: Literal["evidence", "requirement", "claim"]
    target_id: str
    text: str
    status: str
    reason: str
    allowed_actions: list[Decision]
    required: bool
    resolved: bool


class ReviewValidation(BaseModel):
    timestamp: str
    changed_claim_ids: list[str]
    model_claim_ids: list[str]
    final_status: ReviewStatus


class ReviewResponse(BaseModel):
    run_id: str
    scope: Literal["run"]
    final_status: ReviewStatus
    items: list[ReviewItem]
    decisions: list[ReviewDecision]
    pending_required: int
    can_complete: bool
    last_validation: ReviewValidation | None
