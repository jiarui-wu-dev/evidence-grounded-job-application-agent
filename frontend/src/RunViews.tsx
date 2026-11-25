import { Badge } from "./Inspector";
import { CV_SECTIONS } from "./api/types";
import type { Bullet, CV, CVSection, Evidence, EvidenceView, Verdict } from "./api/types";

export type Inspect = (ids: string[], title: string) => void;

export function Requirements({ data, catalog, inspect }: {
  data: EvidenceView | null; catalog: Map<string, Evidence>; inspect: Inspect;
}) {
  const requirements = data?.requirements ?? [];
  if (!requirements.length) return <p className="empty">No requirements have been recorded for this run.</p>;
  return <div className="requirements" aria-label="Requirements">
    {requirements.map((row) => <details key={row.requirement_id} className="requirement"
      onToggle={(event) => {
        if (event.currentTarget.open && row.evidence_ids.length) inspect(row.evidence_ids, row.requirement);
      }}>
      <summary><span>{row.requirement}</span><Badge value={row.review_status ?? row.status ?? "Awaiting match"} /></summary>
      <div className="requirement-detail">
        {row.explanation && <p>{row.explanation}</p>}
        {row.evidence_ids.length ? <ul className="citations">
          {row.evidence_ids.map((id) => {
            const evidence = catalog.get(id);
            return <li key={id}><button className="text-button" onClick={() => inspect([id], row.requirement)}>{id}</button>
              <span className="citation-source">{evidence?.source_id ?? "Source unavailable"}</span>
              <Badge value={evidence?.evidence_status ?? "Status unavailable"} /></li>;
          })}
        </ul> : <p className="muted">No evidence IDs recorded.</p>}
      </div>
    </details>)}
  </div>;
}

function ClaimText({ item, validation, inspect }: { item: Bullet; validation?: Verdict; inspect: Inspect }) {
  return <div className="claim-row">
    {item.evidence_ids.length ? <button className="claim-button" onClick={() => inspect(item.evidence_ids, item.text)}>
      {item.text}
    </button> : <p>{item.text}</p>}
    <div className="claim-meta">
      {validation && <Badge value={validation.status} />}
      {item.evidence_ids.length > 0 && <span className="muted">{item.evidence_ids.length} evidence {item.evidence_ids.length === 1 ? "ID" : "IDs"}</span>}
    </div>
    {validation?.reason && <p className="verdict-reason">{validation.reason}</p>}
  </div>;
}

export function CVView({ cv, data, inspect }: { cv: CV; data: EvidenceView | null; inspect: Inspect }) {
  const claims = data?.claims ?? [];
  return <article className="cv" aria-label="Grounded final CV">
    <section><h3>Profile</h3><ClaimText item={cv.profile} validation={claims.find((claim) => claim.section === "profile")} inspect={inspect} /></section>
    <section><h3>Skills</h3>{cv.skills.map((group) => <div className="skills" key={group.category}>
      <h4>{group.category}</h4>
      <div className="skill-list">{group.items.map((item) => item.evidence_ids.length
        ? <button className="skill-chip" key={item.name} onClick={() => inspect(item.evidence_ids, item.name)}>{item.name}</button>
        : <span className="skill-chip" key={item.name}>{item.name}</span>)}</div>
    </div>)}</section>
    {(Object.entries(CV_SECTIONS) as [CVSection, string][]).map(([key, label]) => <section key={key}><h3>{label}</h3>
      {cv[key].length ? cv[key].map((entry) => <div className="cv-entry" key={entry.source_id}>
        <h4>{entry.title}</h4>{entry.organization && <p className="muted">{entry.organization}</p>}
        <ul>{entry.bullets.map((item, index) => <li key={index}><ClaimText item={item}
          validation={claims.find((claim) => claim.section === key && claim.source_id === entry.source_id && claim.bullet_index === index)}
          inspect={inspect} /></li>)}</ul>
      </div>) : <p className="muted">No entries recorded.</p>}
    </section>)}
    <section><h3>Education</h3>{data?.education.length ? data.education.map((entry) => <div className="cv-entry" key={entry.institution + entry.degree}>
      <h4>{entry.degree}</h4><p>{entry.institution}{entry.location && ` · ${entry.location}`}</p>
      <p className="muted">{[entry.period, ...entry.details].filter(Boolean).join(" · ")}</p>
    </div>) : <p className="muted">Education details are unavailable.</p>}</section>
  </article>;
}

export function Revisions({ data, withholdText }: { data: EvidenceView | null; withholdText: boolean }) {
  const revisions = data?.revisions ?? [];
  if (!revisions.length) return <p className="empty">No changed claims are recorded in the available revision history.</p>;
  return <div className="revision-list">
    {withholdText && <p className="notice">Claim wording is withheld while this run requires review.</p>}
    {revisions.map((item) => <article className="revision" key={`${item.round}:${item.claim_id}`}>
      <div className="revision-heading"><h3>Round {item.round}</h3><code>{item.claim_id}</code></div>
      <div className="revision-comparison">
        <div><h4>Original claim</h4>{!withholdText && item.before_text && <p>{item.before_text}</p>}
          {item.before_validation && <><Badge value={item.before_validation.status} />
            {item.before_validation.reason && <p className="verdict-reason">{item.before_validation.reason}</p>}</>}
        </div>
        <div><h4>Revised claim</h4>{!withholdText && item.after_text && <p>{item.after_text}</p>}
          {item.after_validation ? <><Badge value={item.after_validation.status} />
            {item.after_validation.reason && <p className="verdict-reason">{item.after_validation.reason}</p>}</>
            : <p className="muted">Re-validation verdict unavailable.</p>}
        </div>
      </div>
    </article>)}
  </div>;
}
