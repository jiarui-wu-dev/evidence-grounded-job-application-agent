import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { after, before, test } from "node:test";
import { chromium } from "playwright";

const fixture = JSON.parse(readFileSync(new URL("../src/demo.json", import.meta.url), "utf8"));
const dist = fileURLToPath(new URL("../dist", import.meta.url));
const chrome = process.env.WORKBENCH_CHROME
  ?? (process.platform === "win32" && existsSync("C:/Program Files/Google/Chrome/Application/chrome.exe")
    ? "C:/Program Files/Google/Chrome/Application/chrome.exe" : undefined);
const origin = "http://workbench.test";
const id = "test-run";
const runPath = `/api/runs/${id}`;
const run = (changes = {}) => ({ ...fixture.run, run_id: id, ...changes });
const trace = { ...fixture.trace, run_id: id };
let browser;

before(async () => { browser = await chromium.launch({ executablePath: chrome }); });
after(async () => { await browser?.close(); });

async function open(t, respond = () => undefined) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.setDefaultTimeout(8000);
  t.after(() => page.close());
  const requests = [];
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (!url.pathname.startsWith("/api/")) {
      return route.fulfill({ path: join(dist, url.pathname === "/" ? "index.html" : url.pathname.slice(1)) });
    }
    const request = route.request();
    requests.push({ path: url.pathname, method: request.method(), body: request.postData() });
    const response = respond(url.pathname, request);
    if (response === "offline") return route.abort("failed");
    if (response !== undefined) return route.fulfill({ status: response.status ?? 200, json: response.body });
    const stored = {
      [runPath]: run(), [runPath + "/trace"]: trace,
      [runPath + "/evidence"]: fixture.evidence, [runPath + "/cv"]: fixture.cv,
    };
    return route.fulfill({ status: stored[url.pathname] ? 200 : 404, json: stored[url.pathname] ?? {} });
  });
  await page.goto(origin);
  await page.getByRole("heading", { name: "Application evidence" }).waitFor();
  return { page, requests };
}

async function showLiveRun(page, value = id) {
  await page.getByRole("button", { name: "Live API" }).click();
  await page.getByLabel("Existing run ID").fill(value);
  await page.getByRole("button", { name: "Open run" }).click();
}

test("submits JD and renders a running stage from the backend trace", async (t) => {
  const { page, requests } = await open(t, (path) => {
    if (path === "/api/applications") return { status: 202, body: run({ status: "queued", final_status: null }) };
    if (path === runPath) return { body: run({ status: "running", final_status: null }) };
    if (path === runPath + "/trace") return { body: { ...trace, events: trace.events.slice(0, 3) } };
  });
  await page.getByRole("button", { name: "Live API" }).click();
  await page.getByLabel("Job description", { exact: true }).fill("A realistic JD");
  await page.getByRole("button", { name: "Submit application" }).click();
  await page.locator(".run-id").getByText(id, { exact: true }).waitFor();
  await page.locator("li[data-state=running]").getByText("Retrieval", { exact: true }).waitFor();
  assert.deepEqual(requests.find((item) => item.path === "/api/applications"), {
    path: "/api/applications", method: "POST", body: '{"jd_text":"A realistic JD"}',
  });
});

test("renders all requirement statuses and opens source provenance", async (t) => {
  const provisional = fixture.evidence.evidence.find((item) => item.evidence_status === "provisional");
  const evidence = { ...fixture.evidence, requirements: [...fixture.evidence.requirements, {
    requirement_id: "test_partial", requirement: "Provisional experience", status: "PARTIAL",
    evidence_ids: [provisional.evidence_id], explanation: "This mock row cites provisional evidence.",
  }] };
  const { page } = await open(t, (path) => path === runPath + "/evidence" ? { body: evidence } : undefined);
  await showLiveRun(page);
  const mapping = page.getByLabel("Requirements", { exact: true });
  for (const status of ["SUPPORTED", "PARTIAL", "GAP"]) await mapping.getByText(status, { exact: true }).first().waitFor();
  await mapping.getByText("Provisional experience", { exact: true }).click();
  const inspector = page.getByLabel("Evidence inspector");
  await inspector.getByText(provisional.text, { exact: true }).waitFor();
  await inspector.getByText(provisional.source_id, { exact: true }).waitFor();
  await inspector.getByText("provisional", { exact: true }).waitFor();
});

test("saved demo renders a grounded CV, clickable claims, and recorded revision wording without API calls", async (t) => {
  const { page, requests } = await open(t);
  const summary = page.getByLabel("Recorded benchmark summary");
  await summary.getByText("5/5 roles", { exact: true }).waitFor();
  await summary.getByText("2/2 runs", { exact: true }).waitFor();
  await summary.getByText("76%", { exact: true }).waitFor();
  await summary.getByText("Per-role results", { exact: true }).click();
  await summary.getByText("Backend Software Engineering", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Final CV", exact: true }).click();
  const cv = page.getByLabel("Grounded final CV");
  for (const section of ["Profile", "Skills", "Research Experience", "Professional Experience", "Selected Projects", "Education"]) {
    await cv.getByRole("heading", { name: section, exact: true }).waitFor();
  }
  await cv.getByRole("button", { name: fixture.cv.profile.text, exact: true }).click();
  await page.getByLabel("Evidence inspector").getByText(fixture.cv.profile.evidence_ids[0], { exact: true }).waitFor();
  await cv.getByText(fixture.evidence.claims.find((item) => item.claim_id === "profile").reason, { exact: true }).waitFor();
  await page.getByRole("button", { name: "Revisions", exact: true }).click();
  await page.getByText(fixture.evidence.revisions[0].before_text, { exact: true }).first().waitFor();
  await page.getByText(fixture.evidence.revisions[0].after_text, { exact: true }).first().waitFor();
  assert.deepEqual(requests, []);
});

test("honors CV HTTP 409 and keeps allowed diagnostics visible", async (t) => {
  const { page, requests } = await open(t, (path) => {
    if (path === runPath) return { body: run({ final_status: "needs_human_review" }) };
    if (path === runPath + "/cv") return { status: 409, body: { detail: "raw diagnostic" } };
  });
  await showLiveRun(page);
  await page.getByText(/Human review required\. The final CV/).waitFor();
  await page.getByText("dense retrieval", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Final CV", exact: true }).click();
  await page.getByText("CV access is blocked pending human review.").waitFor();
  assert.equal(await page.getByLabel("Grounded final CV").count(), 0);
  await page.getByRole("button", { name: "Revisions", exact: true }).click();
  await page.getByText("Claim wording is withheld while this run requires review.").waitFor();
  assert.equal(await page.getByText(fixture.evidence.revisions[0].before_text, { exact: true }).count(), 0);
  assert.ok(requests.some((item) => item.path === runPath + "/cv"));
});

test("shows a failed run and resumes it under the same run ID", async (t) => {
  let resumed = false;
  const { page, requests } = await open(t, (path) => {
    if (path === runPath + "/resume") {
      resumed = true;
      return { status: 202, body: run({ status: "queued", final_status: null }) };
    }
    if (path === runPath && !resumed) return { body: run({ status: "failed", final_status: null, error_code: "application_run_failed" }) };
  });
  await showLiveRun(page);
  await page.getByText(/The run failed/).waitFor();
  await page.getByRole("button", { name: "Resume run", exact: true }).click();
  await page.locator(".run-actions").getByText("Grounded", { exact: true }).waitFor();
  assert.ok(requests.some((item) => item.path === runPath + "/resume" && item.method === "POST"));
});

test("shows a paused run and allows checkpoint resume", async (t) => {
  const { page } = await open(t, (path) => path === runPath
    ? { body: run({ status: "failed", final_status: null, error_code: "worker_interrupted" }) } : undefined);
  await showLiveRun(page);
  await page.getByText("Paused", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Resume run" }).waitFor();
});

test("handles invalid JD and unknown run without echoing backend error details", async (t) => {
  const { page, requests } = await open(t, (path) => path === "/api/applications"
    ? { status: 422, body: { detail: "sensitive internal trace" } } : undefined);
  await page.getByRole("button", { name: "Live API" }).click();
  await page.getByRole("button", { name: "Submit application" }).click();
  await page.getByRole("alert").getByText(/Enter a job description/).waitFor();
  assert.equal(requests.length, 0);
  await page.getByLabel("Job description", { exact: true }).fill("Server-rejected JD");
  await Promise.all([
    page.waitForResponse((response) => response.status() === 422),
    page.getByRole("button", { name: "Submit application" }).click(),
  ]);
  assert.equal(await page.getByText("sensitive internal trace").count(), 0);
  await page.getByLabel("Existing run ID").fill("missing");
  await page.getByRole("button", { name: "Open run" }).click();
  await page.getByText(/Run not found/).waitFor();
});

test("shows a friendly error for an unavailable backend", async (t) => {
  const { page } = await open(t, (path) => path === runPath ? "offline" : undefined);
  await showLiveRun(page);
  await page.getByRole("alert").getByText(/backend is unavailable/).waitFor();
});

test("rejects malformed API responses", async (t) => {
  const { page } = await open(t, (path) => path === runPath ? { body: { run_id: id, status: "invalid" } } : undefined);
  await showLiveRun(page);
  await page.getByRole("alert").getByText(/unexpected response/).waitFor();
});

function reviewScenario({ failComplete = false } = {}) {
  const stored = fixture.evidence.evidence.find((item) => item.evidence_status === "provisional");
  const evidenceId = stored.evidence_id;
  const view = {
    run_id: id, scope: "run", final_status: "needs_human_review",
    items: [
      { target_type: "evidence", target_id: evidenceId, text: stored.text, status: "provisional",
        reason: "User review required.", allowed_actions: ["confirm", "reject"], required: true, resolved: false },
      { target_type: "requirement", target_id: "gap_1", text: "Unverified skill", status: "GAP",
        reason: "No evidence recorded.", allowed_actions: ["keep_gap", "ignore_for_application"], required: true, resolved: false },
    ],
    decisions: [], pending_required: 2, can_complete: false, last_validation: null,
  };
  return {
    evidenceId, itemText: stored.text,
    respond(path) {
      if (path === runPath) return { body: run({ final_status: view.final_status }) };
      if (path === runPath + "/cv" && view.final_status !== "grounded") return { status: 409, body: {} };
      if (path === runPath + "/review") return { body: view };
      if (path === runPath + "/review/complete") {
        if (failComplete) return { status: 502, body: { detail: "private provider text" } };
        view.final_status = "grounded";
        view.can_complete = false;
        view.last_validation = { timestamp: "2026-11-24T15:00:00Z", changed_claim_ids: ["profile"],
          model_claim_ids: ["profile"], final_status: "grounded" };
        return { body: view };
      }
    },
    decide(decision) {
      const item = view.items.find((item) => item.allowed_actions.includes(decision));
      const next = { confirm: "verified", reject: "rejected", keep_gap: "GAP", ignore_for_application: "ignored" }[decision];
      view.decisions.push({
        review_id: "review-" + view.decisions.length, run_id: id, target_id: item.target_id,
        target_type: item.target_type, decision, previous_status: item.status, new_status: next,
        timestamp: "2026-11-24T15:00:00Z", note: null,
      });
      item.status = next;
      item.resolved = true;
      item.allowed_actions = [];
      view.pending_required--;
      view.can_complete = view.pending_required === 0;
      return { body: view };
    },
  };
}

test("renders the run-scoped review queue and confirms evidence before unlocking the CV", async (t) => {
  const scenario = reviewScenario();
  const { page, requests } = await open(t, (path, request) => path.includes("/review/evidence/") || path.includes("/review/requirements/")
    ? scenario.decide(request.postDataJSON().decision) : scenario.respond(path));
  await showLiveRun(page);
  await page.getByRole("button", { name: "Human review", exact: true }).click();
  const review = page.getByLabel("Human review", { exact: true });
  await review.getByText(/Decisions apply to this run only/).waitFor();
  const item = review.locator(`[data-review-target="${scenario.evidenceId}"]`);
  await item.getByText("provisional", { exact: true }).waitFor();
  await item.getByRole("button", { name: "Inspect evidence" }).click();
  await page.getByLabel("Evidence inspector").getByText(scenario.itemText, { exact: true }).last().waitFor();
  assert.equal(await review.getByRole("button", { name: "Complete review" }).isDisabled(), true);
  await item.getByRole("button", { name: "Confirm evidence" }).click();
  await item.getByText("verified", { exact: true }).waitFor();
  await review.getByRole("button", { name: "Keep as GAP" }).click();
  await review.getByRole("heading", { name: "Decision history" }).waitFor();
  await review.getByRole("button", { name: "Complete review" }).click();
  await review.getByText(/Review complete\. The grounded CV/).waitFor();
  await page.getByRole("button", { name: "Final CV", exact: true }).click();
  await page.getByLabel("Grounded final CV").waitFor();
  assert.ok(requests.some((request) => request.body === '{"decision":"confirm"}'));
});

test("records rejection and ignored GAP, then reports validation failure without exposing backend detail", async (t) => {
  const scenario = reviewScenario({ failComplete: true });
  const { page, requests } = await open(t, (path, request) => path.includes("/review/evidence/") || path.includes("/review/requirements/")
    ? scenario.decide(request.postDataJSON().decision) : scenario.respond(path));
  await showLiveRun(page);
  await page.getByRole("button", { name: "Human review", exact: true }).click();
  const review = page.getByLabel("Human review", { exact: true });
  const item = review.locator(`[data-review-target="${scenario.evidenceId}"]`);
  await item.getByRole("button", { name: "Reject evidence" }).click();
  await item.getByText("rejected", { exact: true }).waitFor();
  await review.getByRole("button", { name: "Ignore for this application" }).click();
  await review.locator('[data-review-target="gap_1"]').getByText("ignored", { exact: true }).waitFor();
  await review.getByRole("button", { name: "Complete review" }).click();
  await review.getByRole("alert").getByText(/saved decisions are intact/).waitFor();
  assert.equal(await page.getByText("private provider text", { exact: true }).count(), 0);
  await page.getByRole("button", { name: "Final CV", exact: true }).click();
  await page.getByText("CV access is blocked pending human review.").waitFor();
  assert.ok(requests.some((request) => request.body === '{"decision":"reject"}'));
});
