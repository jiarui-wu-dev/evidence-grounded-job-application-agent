export type RunStatus = "queued" | "running" | "completed" | "failed";
export type FinalStatus = "grounded" | "needs_human_review";
export type RequirementStatus = "SUPPORTED" | "PARTIAL" | "GAP";
export type ClaimStatus = "SUPPORTED" | "OVERSTATED" | "UNSUPPORTED";

export interface Run {
  run_id: string;
  status: RunStatus;
  final_status: FinalStatus | null;
  error_code: string | null;
  created_at: string;
  updated_at: string;
}

export interface TraceEvent {
  created_at: string | null;
  next_nodes: string[];
  revision_round: number | null;
  pending_revalidation_claim_ids: string[] | null;
  validation_counts: Record<string, number>;
  final_status: FinalStatus | null;
}

export interface Trace { run_id: string; events: TraceEvent[] }

export interface Evidence {
  evidence_id: string;
  source_id: string | null;
  source_type: string | null;
  text: string;
  evidence_status: string | null;
}

export interface Requirement {
  requirement_id: string;
  requirement: string;
  status: RequirementStatus | null;
  explanation: string | null;
  evidence_ids: string[];
  review_status?: "GAP" | "ignored" | null;
}

export interface Verdict { status: ClaimStatus; reason: string | null }

export interface Claim extends Verdict {
  claim_id: string;
  section: string | null;
  source_id: string | null;
  bullet_index: number | null;
}

export interface Revision {
  round: number;
  claim_id: string;
  before_text: string | null;
  after_text: string | null;
  before_validation: Verdict | null;
  after_validation: Verdict | null;
}

export interface Education {
  institution: string;
  degree: string;
  location: string;
  period: string;
  details: string[];
}

export interface EvidenceView {
  requirements: Requirement[];
  evidence: Evidence[];
  claims: Claim[];
  revisions: Revision[];
  education: Education[];
}

export interface Bullet { text: string; evidence_ids: string[] }
export interface CVEntry { source_id: string; title: string; organization?: string; bullets: Bullet[] }
export interface SkillGroup { category: string; items: { name: string; evidence_ids: string[] }[] }
export const CV_SECTIONS = {
  research_experience: "Research Experience",
  experience: "Professional Experience",
  selected_projects: "Selected Projects",
} as const;
export type CVSection = keyof typeof CV_SECTIONS;
export interface CV extends Record<CVSection, CVEntry[]> {
  profile: Bullet;
  skills: SkillGroup[];
  job_title?: string;
}

export type EvidenceDecision = "confirm" | "reject";
export type RequirementDecision = "keep_gap" | "ignore_for_application";
export type Decision = EvidenceDecision | RequirementDecision;
export interface ReviewItem {
  target_type: "evidence" | "requirement" | "claim";
  target_id: string;
  text: string;
  status: string;
  reason: string;
  allowed_actions: Decision[];
  required: boolean;
  resolved: boolean;
}
export interface ReviewAudit {
  review_id: string;
  run_id: string;
  target_type: "evidence" | "requirement";
  target_id: string;
  decision: Decision;
  previous_status: string;
  new_status: string;
  timestamp: string;
  note: string | null;
}
export interface Review {
  run_id: string;
  scope: "run";
  final_status: FinalStatus;
  items: ReviewItem[];
  decisions: ReviewAudit[];
  pending_required: number;
  can_complete: boolean;
  last_validation: {
    timestamp: string;
    changed_claim_ids: string[];
    model_claim_ids: string[];
    final_status: FinalStatus;
  } | null;
}
