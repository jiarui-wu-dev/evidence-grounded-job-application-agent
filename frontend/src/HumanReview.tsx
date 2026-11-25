import { useEffect, useState } from "react";
import { ApiError, api, errorMessage } from "./api/client";
import type { Decision, Review } from "./api/types";
import { Badge } from "./Inspector";
import type { Inspect } from "./RunViews";

const labels: Record<Decision, string> = {
  confirm: "Confirm evidence", reject: "Reject evidence",
  keep_gap: "Keep as GAP", ignore_for_application: "Ignore for this application",
};

export function HumanReview({ runID, inspect, onChange }: {
  runID: string; inspect: Inspect; onChange: () => void;
}) {
  const [view, setView] = useState<Review | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  function verify(response: Review) {
    if (response.run_id !== runID) throw new ApiError("malformed");
    return response;
  }

  useEffect(() => {
    let current = true;
    api.getReview(runID).then((response) => {
      if (current) setView(verify(response));
    }).catch((failure) => {
      if (current) setError(errorMessage(failure));
    });
    return () => { current = false; };
  }, [runID, retry]);

  async function act(request: () => Promise<Review>) {
    setBusy(true);
    setError("");
    try {
      setView(verify(await request()));
      onChange();
    } catch (failure) {
      setError(failure instanceof ApiError && failure.status === 502
        ? "Review validation failed. Your saved decisions are intact. Try completing review again."
        : errorMessage(failure));
    } finally {
      setBusy(false);
    }
  }

  return <section aria-label="Human review">
    <div className="content-heading"><h2>Human review</h2>{view?.pending_required ? <span className="muted">{view.pending_required} required</span> : null}</div>
    <p className="review-scope">Decisions apply to this run only. The Candidate KB keeps its original evidence status.</p>
    {error && <div className="error review-error" role="alert">{error} {!view &&
      <button onClick={() => { setError(""); setRetry((count) => count + 1); }}>Retry</button>}</div>}
    {!view ? !error && <p className="empty">Loading review queue…</p> : <>
      {view.items.length === 0 && <p className="empty">No review items are recorded for this run.</p>}
      <div className="review-queue">{view.items.map((item) => <article className="review-item" key={`${item.target_type}:${item.target_id}`} data-review-target={item.target_id}>
        <header><h3>{item.target_type === "evidence" ? "Evidence" : item.target_type === "requirement" ? "Requirement" : "Flagged claim"}</h3>
          <Badge value={item.status} /></header>
        <p>{item.text}</p>
        <p className="verdict-reason">{item.reason}</p>
        <div className="review-actions">
          {item.target_type === "evidence" && <button type="button" onClick={() => inspect([item.target_id], item.text)}>Inspect evidence</button>}
          {item.allowed_actions.map((action) => <button type="button" key={action} disabled={busy} onClick={() => {
            if (action === "confirm" || action === "reject") {
              void act(() => api.decideEvidence(runID, item.target_id, action));
            } else {
              void act(() => api.decideRequirement(runID, item.target_id, action));
            }
          }}>{labels[action]}</button>)}
        </div>
      </article>)}</div>
      {view.final_status === "needs_human_review" && <div className="review-complete">
        <button className="primary" type="button" disabled={busy || !view.can_complete}
          onClick={() => act(() => api.completeReview(runID))}>Complete review</button>
        {view.pending_required > 0 && <p className="muted">Resolve required items before completing review.</p>}
        {view.last_validation && <p className="muted" role="status">The CV still has unresolved claims. Export is blocked.</p>}
      </div>}
      {view.final_status === "grounded" && view.last_validation && <p className="review-scope" role="status">Review complete. The grounded CV is available.</p>}
      {view.last_validation && <details className="review-checks">
        <summary>Review validation results</summary>
        <p>{view.last_validation.model_claim_ids.length} affected claims sent to the Validator.</p>
        {view.last_validation.changed_claim_ids.map((id) => <p key={id}><code>{id}</code></p>)}
      </details>}
      {view.decisions.length > 0 && <div className="review-history"><h3>Decision history</h3>
        <ul>{view.decisions.map((row) => <li key={row.review_id}>
          <strong>{labels[row.decision]}</strong> <code>{row.target_id}</code>
          <p><Badge value={row.previous_status} /> → <Badge value={row.new_status} />
            <time dateTime={row.timestamp}>{new Date(row.timestamp).toLocaleString()}</time></p>
          {row.note && <p>{row.note}</p>}
        </li>)}</ul>
      </div>}
    </>}
  </section>;
}
