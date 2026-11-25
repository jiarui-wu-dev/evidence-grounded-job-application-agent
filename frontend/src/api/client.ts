import { CV_SECTIONS } from "./types";
import type { CV, EvidenceDecision, EvidenceView, RequirementDecision, Review, Run, Trace } from "./types";

type Item = Record<string, unknown>;
const object = (v: unknown): v is Item => typeof v === "object" && v !== null && !Array.isArray(v);
const string = (v: unknown): v is string => typeof v === "string";
const integer = (v: unknown): v is number => Number.isInteger(v);
const nullable = (v: unknown, check: (value: unknown) => boolean) => v === null || check(v);
const list = (v: unknown, check: (value: unknown) => boolean) => Array.isArray(v) && v.every(check);
const strings = (v: unknown) => list(v, string);
const oneOf = (v: unknown, values: string[]) => string(v) && values.includes(v);
const final = (v: unknown) => nullable(v, (x) => oneOf(x, ["grounded", "needs_human_review"]));

function isRun(v: unknown): v is Run {
  return object(v) && string(v.run_id) && oneOf(v.status, ["queued", "running", "completed", "failed"])
    && final(v.final_status) && nullable(v.error_code, string) && string(v.created_at) && string(v.updated_at);
}

function isTrace(v: unknown): v is Trace {
  return object(v) && string(v.run_id) && list(v.events, (e) => object(e)
    && nullable(e.created_at, string) && strings(e.next_nodes) && nullable(e.revision_round, integer)
    && nullable(e.pending_revalidation_claim_ids, strings) && final(e.final_status)
    && object(e.validation_counts) && Object.values(e.validation_counts).every(integer));
}

const verdict = (v: unknown) => object(v) && oneOf(v.status, ["SUPPORTED", "OVERSTATED", "UNSUPPORTED"])
  && nullable(v.reason, string);

function isEvidenceView(v: unknown): v is EvidenceView {
  return object(v)
    && list(v.requirements, (r) => object(r) && string(r.requirement_id) && string(r.requirement)
      && nullable(r.status, (x) => oneOf(x, ["SUPPORTED", "PARTIAL", "GAP"]))
      && (r.review_status === undefined || nullable(r.review_status, (x) => oneOf(x, ["GAP", "ignored"])))
      && nullable(r.explanation, string) && strings(r.evidence_ids))
    && list(v.evidence, (e) => object(e) && string(e.evidence_id) && string(e.text)
      && nullable(e.source_id, string) && nullable(e.source_type, string) && nullable(e.evidence_status, string))
    && list(v.claims, (c) => object(c) && verdict(c) && string(c.claim_id) && nullable(c.section, string)
      && nullable(c.source_id, string) && nullable(c.bullet_index, integer))
    && list(v.revisions, (r) => object(r) && integer(r.round) && string(r.claim_id)
      && nullable(r.before_text, string) && nullable(r.after_text, string)
      && nullable(r.before_validation, verdict) && nullable(r.after_validation, verdict))
    && list(v.education, (e) => object(e) && [e.degree, e.institution, e.location, e.period].every(string)
      && strings(e.details));
}

const bullet = (v: unknown) => object(v) && string(v.text) && strings(v.evidence_ids);

function isCV(v: unknown): v is CV {
  return object(v) && bullet(v.profile) && (v.job_title === undefined || string(v.job_title))
    && list(v.skills, (g) => object(g) && string(g.category) && list(g.items,
      (s) => object(s) && string(s.name) && strings(s.evidence_ids)))
    && Object.keys(CV_SECTIONS).every((section) => list(v[section], (e) => object(e)
      && string(e.source_id) && string(e.title) && (e.organization === undefined || string(e.organization))
      && list(e.bullets, bullet)));
}

const knownDecision = (v: unknown) => oneOf(v, ["confirm", "reject", "keep_gap", "ignore_for_application"]);
const reviewFinal = (v: unknown) => v !== null && final(v);
function isReview(v: unknown): v is Review {
  return object(v) && string(v.run_id) && v.scope === "run" && reviewFinal(v.final_status)
    && integer(v.pending_required) && typeof v.can_complete === "boolean"
    && list(v.items, (item) => object(item)
      && oneOf(item.target_type, ["evidence", "requirement", "claim"])
      && [item.target_id, item.text, item.status, item.reason].every(string)
      && typeof item.required === "boolean" && typeof item.resolved === "boolean"
      && list(item.allowed_actions, knownDecision))
    && list(v.decisions, (row) => object(row) && oneOf(row.target_type, ["evidence", "requirement"])
      && [row.review_id, row.run_id, row.target_id, row.previous_status, row.new_status, row.timestamp].every(string)
      && knownDecision(row.decision) && nullable(row.note, string))
    && nullable(v.last_validation, (row) => object(row) && string(row.timestamp)
      && strings(row.changed_claim_ids) && strings(row.model_claim_ids) && reviewFinal(row.final_status));
}

export class ApiError extends Error {
  constructor(readonly kind: "offline" | "malformed" | "http", readonly status?: number) {
    super(kind);
  }
}

async function request<T>(path: string, check: (value: unknown) => value is T, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError("offline");
  }
  if (!response.ok) throw new ApiError("http", response.status);
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new ApiError("malformed");
  }
  if (!check(value)) throw new ApiError("malformed");
  return value;
}

const endpoint = (id: string, suffix = "") => `/runs/${encodeURIComponent(id)}${suffix}`;
const decisionRequest = (decision: EvidenceDecision | RequirementDecision): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision }),
});
export const api = {
  createApplication: (jd_text: string) => request("/applications", isRun, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ jd_text }),
  }),
  getRun: (id: string) => request(endpoint(id), isRun),
  getTrace: (id: string) => request(endpoint(id, "/trace"), isTrace),
  getEvidence: (id: string) => request(endpoint(id, "/evidence"), isEvidenceView),
  getCV: (id: string) => request(endpoint(id, "/cv"), isCV),
  resumeRun: (id: string) => request(endpoint(id, "/resume"), isRun, { method: "POST" }),
  getReview: (id: string) => request(endpoint(id, "/review"), isReview),
  decideEvidence: (id: string, evidenceId: string, decision: EvidenceDecision) =>
    request(endpoint(id, `/review/evidence/${encodeURIComponent(evidenceId)}`), isReview, decisionRequest(decision)),
  decideRequirement: (id: string, requirementId: string, decision: RequirementDecision) =>
    request(endpoint(id, `/review/requirements/${encodeURIComponent(requirementId)}`), isReview, decisionRequest(decision)),
  completeReview: (id: string) => request(endpoint(id, "/review/complete"), isReview, { method: "POST" }),
};

export function errorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return "The request could not be completed.";
  if (error.kind === "offline" || [502, 503, 504].includes(error.status ?? 0)) {
    return "The backend is unavailable. Check the local FastAPI service and try again.";
  }
  if (error.kind === "malformed") return "The API returned an unexpected response.";
  if (error.status === 422) return "Enter a job description between 1 and 30,000 characters.";
  if (error.status === 404) return "Run not found. Check the run ID.";
  if (error.status === 409) return "That action is unavailable for the current run state.";
  return "The API could not complete the request.";
}
