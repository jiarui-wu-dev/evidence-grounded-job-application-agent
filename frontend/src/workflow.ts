import type { Run, TraceEvent } from "./api/types";

const stages = [
  ["analyze_jd", "JD analysis"], ["build_queries", "Query building"], ["retrieve_evidence", "Retrieval"],
  ["analyze_match_gap", "Match / gap"], ["build_strategy", "Strategy"], ["expand_sources", "Composition"],
  ["tailor_cv", "CV generation"], ["validate_cv", "Validation"], ["revise_cv", "Targeted revision"],
  ["revalidate_cv", "Targeted re-validation"], ["finish", "Finalization"],
] as const;

const eventNodes = (event: TraceEvent) => event.next_nodes.map((node) => {
  if (node === "validate_cv" && (event.revision_round ?? 0) > 0) return "revalidate_cv";
  if (node === "finalize" || node === "finalize_with_warning") return "finish";
  return node;
});

export function pipeline(run: Run | null, events: TraceEvent[]) {
  const completed = new Set(events.slice(0, -1).flatMap(eventNodes));
  const latest = events.at(-1);
  const current = new Set(latest ? eventNodes(latest) : []);
  return stages.map(([id, label]) => {
    const state = id === "finish" && run?.final_status === "needs_human_review" ? "warning"
      : current.has(id) && run?.status === "failed" ? "failed"
      : current.has(id) && run?.status === "running" ? "running"
      : completed.has(id) ? "completed"
      : run?.status === "completed" && latest?.final_status ? "not taken"
      : "pending";
    return { id, label, state };
  });
}
