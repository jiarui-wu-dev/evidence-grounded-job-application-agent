"""Public response schema for the read-only evidence inspector."""

from typing import Literal

from pydantic import BaseModel


class Requirement(BaseModel):
    requirement_id: str
    requirement: str
    status: Literal["SUPPORTED", "PARTIAL", "GAP"] | None
    explanation: str | None
    evidence_ids: list[str]
    review_status: Literal["GAP", "ignored"] | None = None


class Evidence(BaseModel):
    evidence_id: str
    source_id: str | None
    source_type: str | None
    text: str
    evidence_status: str | None


class Verdict(BaseModel):
    status: Literal["SUPPORTED", "OVERSTATED", "UNSUPPORTED"]
    reason: str | None


class Claim(Verdict):
    claim_id: str
    section: str | None
    source_id: str | None
    bullet_index: int | None


class Revision(BaseModel):
    round: int
    claim_id: str
    before_text: str | None
    after_text: str | None
    before_validation: Verdict | None
    after_validation: Verdict | None


class Education(BaseModel):
    institution: str
    degree: str
    location: str
    period: str
    details: list[str]


class EvidenceResponse(BaseModel):
    requirements: list[Requirement]
    evidence: list[Evidence]
    claims: list[Claim]
    revisions: list[Revision]
    education: list[Education]
