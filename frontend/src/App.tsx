import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { api, ApiError, errorMessage } from "./api/client";
import type { CV, EvidenceView, Run, Trace } from "./api/types";
import { Badge, EvidenceInspector } from "./Inspector";
import { CVView, Requirements, Revisions } from "./RunViews";
import { HumanReview } from "./HumanReview";
import { pipeline } from "./workflow";
import savedDemo from "./demo.json";
import { BenchmarkSummary } from "./BenchmarkSummary";

const demo = savedDemo as { jd_text: string; run: Run; trace: Trace; evidence: EvidenceView; cv: CV };
type Tab = "Requirements" | "Final CV" | "Revisions" | "Human review";
const tabs: Tab[] = ["Requirements", "Final CV", "Revisions"];
const demoOnly = import.meta.env.VITE_DEMO_ONLY === "1";

export function App() {
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [jd, setJD] = useState("");
  const [openID, setOpenID] = useState("");
  const [runID, setRunID] = useState("");
  const [liveRun, setLiveRun] = useState<Run | null>(null);
  const [liveTrace, setLiveTrace] = useState<Trace | null>(null);
  const [liveEvidence, setLiveEvidence] = useState<EvidenceView | null>(null);
  const [liveCV, setLiveCV] = useState<CV | null>(null);
  const [tab, setTab] = useState<Tab>("Requirements");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [pollError, setPollError] = useState("");
  const [cvBlocked, setCVBlocked] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [selection, setSelection] = useState({ ids: [] as string[], title: "" });

  const live = mode === "live";
  const run = live ? liveRun : demo.run;
  const trace = live ? liveTrace : demo.trace;
  const evidence = live ? liveEvidence : demo.evidence;
  const cv = live ? liveCV : demo.cv;
  const review = run?.final_status === "needs_human_review" || (live && cvBlocked);
  const visibleTabs: Tab[] = live && run?.status === "completed" ? [...tabs, "Human review"] : tabs;
  const catalog = useMemo(() => new Map(evidence?.evidence.map((item) => [item.evidence_id, item])), [evidence]);
  const stages = pipeline(run, trace?.events ?? []);

  useEffect(() => {
    if (!live || !runID) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function read() {
      let retry = true;
      try {
        const current = await api.getRun(runID);
        if (!active) return;
        if (current.run_id !== runID) throw new ApiError("malformed");
        setLiveRun(current);
        retry = current.status === "queued" || current.status === "running";
        const [nextTrace, nextEvidence] = await Promise.allSettled([api.getTrace(runID), api.getEvidence(runID)]);
        if (!active) return;
        const messages: string[] = [];
        if (nextTrace.status === "fulfilled" && nextTrace.value.run_id === runID) setLiveTrace(nextTrace.value);
        else messages.push("Workflow trace unavailable. " + errorMessage(
          nextTrace.status === "rejected" ? nextTrace.reason : new ApiError("malformed"),
        ));
        if (nextEvidence.status === "fulfilled") setLiveEvidence(nextEvidence.value);
        else messages.push("Evidence details unavailable. " + errorMessage(nextEvidence.reason));
        if (current.status === "completed") {
          try {
            const finalCV = await api.getCV(runID);
            if (active) {
              setLiveCV(finalCV);
              setCVBlocked(false);
            }
          } catch (error) {
            if (!active) return;
            setLiveCV(null);
            if (error instanceof ApiError && error.status === 409) setCVBlocked(true);
            else messages.push(errorMessage(error));
          }
        }
        if (active) setPollError(messages.join(" "));
      } catch (error) {
        if (!active) return;
        setPollError(errorMessage(error));
        if (error instanceof ApiError && error.status === 404) retry = false;
      }
      if (active && retry) timer = setTimeout(read, 1250);
    }
    void read();
    return () => { active = false; clearTimeout(timer); };
  }, [live, runID, refresh]);

  function resetRun(id: string, initial: Run | null = null) {
    setRunID(id);
    setLiveRun(initial);
    setLiveTrace(null);
    setLiveEvidence(null);
    setLiveCV(null);
    setCVBlocked(false);
    setSelection({ ids: [], title: "" });
    setTab("Requirements");
    setPollError("");
    setActionError("");
    setRefresh((count) => count + 1);
  }

  function changeMode(nextMode: "demo" | "live") {
    if (mode === nextMode) return;
    resetRun("");
    setMode(nextMode);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!live || busy) return;
    if (!jd.trim()) {
      setActionError("Enter a job description between 1 and 30,000 characters.");
      return;
    }
    setBusy(true);
    setActionError("");
    try {
      const created = await api.createApplication(jd.trim());
      resetRun(created.run_id, created);
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function resume() {
    if (!run || !live || busy) return;
    setBusy(true);
    setActionError("");
    try {
      const resumed = await api.resumeRun(run.run_id);
      setLiveRun(resumed);
      setRefresh((count) => count + 1);
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  const inspect = (ids: string[], title: string) => setSelection({ ids, title });
  const mapped = evidence?.requirements.filter((item) => item.status !== null).length ?? 0;
  const latest = trace?.events.at(-1);
  const visibleStatus = review ? "Needs human review"
    : run?.final_status === "grounded" ? "Grounded"
    : run?.status === "failed" && run.error_code === "worker_interrupted" ? "Paused"
    : run?.status ?? "No active run";

  return <>
    <header className="topbar">
      <div className="brand"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 28 28" fill="none"><path d="m6 9 8-4 8 4-8 4-8-4Zm0 5 8 4 8-4M6 19l8 4 8-4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/></svg></span>Application Workbench</div>
      <div className="switcher" aria-label="Data source">
        <button aria-pressed={!live} disabled={busy} onClick={() => changeMode("demo")}>Saved demo</button>
        {!demoOnly && <button aria-pressed={live} disabled={busy} onClick={() => changeMode("live")}>Live API</button>}
      </div>
    </header>
    <main>
      <div className="page-intro">
        <div><h1>Application evidence</h1>
          <p>Trace job requirements from retrieved evidence to validated CV claims.</p></div>
        {!live && <span className="demo-label"><span aria-hidden="true" />Saved benchmark run</span>}
      </div>
      {!live && <BenchmarkSummary benchmark={savedDemo.benchmark} />}
      <div className="workspace">
        <div className="left-column">
          <section className="panel intake">
            <div className="section-heading"><h2>Job description</h2></div>
            <form onSubmit={submit} className="intake-body">
              <label htmlFor="jd" className="sr-only">Job description</label>
              <textarea id="jd" value={live ? jd : demo.jd_text} readOnly={!live} maxLength={30000}
                placeholder="Paste the role description…" onChange={(event) => setJD(event.target.value)} rows={6} />
              {live && <button className="primary" type="submit" disabled={busy}>Submit application</button>}
            </form>
            {live && <form className="open-run" onSubmit={(event) => {
              event.preventDefault();
              if (openID.trim()) resetRun(openID.trim());
            }}><label className="sr-only" htmlFor="open-id">Existing run ID</label>
              <input id="open-id" placeholder="Existing run ID" value={openID} onChange={(event) => setOpenID(event.target.value)} />
              <button type="submit" disabled={busy}>Open run</button>
            </form>}
          </section>
          <section className="panel pipeline">
            <div className="section-heading"><h2>Workflow</h2></div>
            <ol aria-label="Workflow stages">{stages.map((stage) => <li key={stage.id} data-state={stage.state}>
              <span className="stage-dot" aria-hidden="true">{stage.state === "completed" ? "✓" : stage.state === "warning" ? "!" : ""}</span>
              <span>{stage.label}</span><small>{stage.state}</small>
            </li>)}</ol>
          </section>
        </div>
        <section className="panel run-panel" aria-label="Run workspace">
          <div className="run-heading">
            <div><h2>{cv?.job_title || "Application run"}</h2>
              {run && <p className="run-id">Run ID <code>{run.run_id}</code></p>}
            </div>
            <div aria-live="polite" className="run-actions"><Badge value={visibleStatus} />
              {live && run && (run.status === "failed" || run.status === "queued") &&
                <button type="button" disabled={busy} onClick={resume}>Resume run</button>}
            </div>
          </div>
          {actionError && <p className="error" role="alert">{actionError}</p>}
          {pollError && <div className="error" role="alert">{pollError} <button onClick={() => setRefresh((count) => count + 1)}>Retry</button></div>}
          {run?.status === "failed" && <p className="notice" role="status">
            {run.error_code === "worker_interrupted" ? "The worker was interrupted. Resume from the checkpoint."
              : "The run failed. Resume to retry from the last saved checkpoint."}</p>}
          {review && <p className="notice" role="status">Human review required. The final CV is unavailable (HTTP 409). Requirement and validation diagnostics are still available.</p>}
          <nav className="tabs" aria-label="Run views">{visibleTabs.map((item) => <button key={item} aria-pressed={tab === item}
            onClick={() => setTab(item)}>{item}</button>)}</nav>
          <div className="run-content">
            {tab === "Requirements" && <>
              <div className="content-heading"><h2>Requirement mapping</h2>{mapped > 0 && <span className="muted">{mapped} mapped</span>}</div>
              <Requirements data={evidence} catalog={catalog} inspect={inspect} />
            </>}
            {tab === "Final CV" && <><div className="content-heading"><h2>Final CV</h2></div>
              {review ? <p className="empty">CV access is blocked pending human review.</p>
                : cv ? <CVView cv={cv} data={evidence} inspect={inspect} />
                : <p className="empty">A grounded final CV is not available yet.</p>}
            </>}
            {tab === "Revisions" && <><div className="content-heading"><h2>Validation and revision</h2></div>
              {latest && Object.keys(latest.validation_counts).length > 0 && <div className="validation-counts">
                {Object.entries(latest.validation_counts).map(([status, count]) => <span key={status}><Badge value={status} /> <b>{count}</b></span>)}
              </div>}
              <Revisions data={evidence} withholdText={review} />
            </>}
            {tab === "Human review" && live && run && <HumanReview key={run.run_id} runID={run.run_id}
              inspect={inspect} onChange={() => setRefresh((count) => count + 1)} />}
          </div>
        </section>
        <EvidenceInspector {...selection} catalog={catalog} />
      </div>
    </main>
  </>;
}
