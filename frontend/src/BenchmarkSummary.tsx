import type demo from "./demo.json";

const percent = (value: number) => new Intl.NumberFormat("en", { style: "percent" }).format(value);

export function BenchmarkSummary({ benchmark }: { benchmark: typeof demo.benchmark }) {
  const stageB = benchmark.roles.filter((row) => row.final_status !== null);
  return <section className="benchmark panel" aria-label="Recorded benchmark summary">
    <div>
      <h2>Recorded benchmark</h2>
      <p className="muted">Saved Phase 1B results using {benchmark.configured_model}. Candidate details are redacted.</p>
    </div>
    <dl>
      <div><dt>Stage A</dt><dd>{benchmark.stage_a_completed}/{benchmark.roles.length} roles</dd></div>
      <div><dt>Stage B grounded</dt><dd>{stageB.filter((row) => row.final_status === "grounded").length}/{benchmark.stage_b_completed} runs</dd></div>
      <div><dt>Scorable requirements</dt><dd>{benchmark.scored_requirements}/{benchmark.gold_requirements}</dd></div>
      <div><dt>Classification macro mean</dt><dd>{percent(benchmark.accuracy)}</dd></div>
    </dl>
    <details><summary>Per-role results</summary>
      <ul>{benchmark.roles.map((row) => <li key={row.role}>
        <span>{row.role}</span> <b>{percent(row.accuracy)}</b>{row.final_status && <small>{row.final_status}</small>}
      </li>)}</ul>
    </details>
  </section>;
}
