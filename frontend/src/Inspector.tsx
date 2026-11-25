import type { Evidence } from "./api/types";

export function Badge({ value }: { value: string }) {
  const style = value.toLowerCase().replaceAll(" ", "-");
  return <span className={`badge badge-${style}`}>{value}</span>;
}

export function EvidenceInspector({ ids, catalog, title }: {
  ids: string[]; catalog: Map<string, Evidence>; title: string;
}) {
  return <aside className="panel inspector" aria-label="Evidence inspector">
    <div className="section-heading"><h2>Evidence inspector</h2></div>
    <div className="inspector-body">
      {ids.length === 0 ? <p className="empty">Open a requirement or select a CV claim to inspect its cited evidence.</p>
        : <><p className="selection-label">{title}</p>
          {ids.map((id) => {
            const item = catalog.get(id);
            return <article className="evidence-item" key={id}>
              <div className="badge-row"><Badge value={item?.evidence_status ?? "Status unavailable"} /></div>
              {item ? <><p>{item.text}</p><dl>
                <dt>Evidence ID</dt><dd>{item.evidence_id}</dd>
                <dt>Source ID</dt><dd>{item.source_id ?? "Unavailable"}</dd>
                <dt>Source type</dt><dd>{item.source_type ?? "Unavailable"}</dd>
              </dl></> : <p>Evidence details are unavailable for <code>{id}</code>.</p>}
            </article>;
          })}
        </>}
    </div>
  </aside>;
}
