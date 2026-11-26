# Shared GitLab HTTP Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all in-process GitLab jobs share pooled connections and throttle
coordination, with one bounded, write-safe retry implementation.

**Architecture:** An application-owned `GitLabHttpRuntime` holds lazy
API-URL/token-keyed clients and request gates. A scoped ContextVar binding connects
existing provider/fetcher/auth callers to that runtime without changing other
forge interfaces. All GitLab HTTP paths use one request executor.

**Tech Stack:** Python 3.11+, asyncio, HTTPX, FastAPI, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-26-gitlab-http-design.md`

## Global Constraints

- Five total attempts per logical HTTP request.
- Overall deadline: 300 seconds, covering queueing, attempts and sleeps.
- Pool and active-request limits: eight each per API URL/token; eight keep-alive
  connections, 30-second keep-alive expiry.
- HTTPX connect/write/pool timeouts: 10/30/5 seconds; read timeout: 30 seconds.
  Preserve the archive caller's 120-second read timeout within the same deadline.
- GET/HEAD retry statuses: 408, 429, 500, 502, 503, 504.
- Other methods retry only an explicit 429 with a replayable payload.
- No server-directed delay is capped downward.
- No cluster support, Redis, new dependencies, live deployment or Ansible edits.
- Preserve non-GitLab behavior and existing GitLab tree validation.
- Work from current fork `origin/main`, not by stacking another retry loop on
  PR 3. Retain its five regression scenarios but replace the duplicated design.

## Review Focus

1. ASGI lifespan state does not automatically reach request ContextVars: Task 4
   must test requests and dashboard-spawned jobs against the actual app runtime.
2. A final-attempt 429 must still delay sibling jobs: Task 2 must test this race.
3. A POST may have committed before a timeout/503: Task 2 must assert one send,
   even with `Retry-After`, while recording applicable shared cooldown.
4. Credential-bearing Link/redirect targets can escape the API: Task 3 must test
   rejection without sending a token to the target.
5. Temporary source failures must not authorize thread resolution or become
   cached absence: Task 5 must prove both behaviors.

## Code map

Create:
- `src/mira/platforms/gitlab/http.py`: ownership, retry policy and shared gates.
- `tests/test_gitlab_http.py`: policy/concurrency/runtime tests.
- `tests/test_gitlab_http_lifecycle.py`: ASGI/CLI ownership tests.
- `tests/test_gitlab_http_pool.py`: loopback keep-alive integration test.

Modify:
- `src/mira/providers/gitlab.py`: replace low-level client calls with the executor.
- `src/mira/platforms/fetch.py`: GitLabRepoFetcher paths only.
- `src/mira/platforms/gitlab/webhook.py`: discovery HTTP and its lifecycle binding.
- `src/mira/platforms/gitlab/auth.py`: identity HTTP.
- `src/mira/platforms/server.py`: runtime lifespan and request middleware binding.
- `src/mira/cli.py`: invocation-scoped runtime inside relevant async entry points.
- `src/mira/index/context.py`: visible, non-cached typed source failures.
- `src/mira/core/threads.py`: skip resolution for unavailable source.
- Existing corresponding provider/fetcher/auth/context/engine/CLI tests.

Do not modify `BaseProvider` or the provider/fetcher factory signatures.

---

### Task 1: Define retry policy and server-directed delay parsing

**Files:** Create `src/mira/platforms/gitlab/http.py`,
`tests/test_gitlab_http.py`.

**Interfaces produced:**
- `GitLabRetryPolicy` frozen dataclass with `max_attempts=5`,
  `deadline_seconds=300.0`, `backoff_cap_seconds=60.0`.
- `parse_retry_after(value: str | None, *, now: datetime) -> float | None`.
- `retryable_status(method: str, status: int) -> bool`.
- `retryable_exception(method: str, exc: httpx.TransportError) -> bool`.
- `GitLabRequestError(ProviderError)` with `reason: str`, `attempts: int`,
  `status_code: int | None`.

- [ ] Write parametrized tests `test_retryable_status` asserting the complete
  spec matrix: GET/HEAD transient statuses true; all write methods true only
  for 429; 401/403/404/409/422/501/505 and 3xx false. Lowercase methods normalize.
- [ ] Write `test_parse_retry_after`: with fixed aware UTC time, `"120"` returns
  120, `"0.5"` returns 0.5, a date 120 seconds ahead returns 120, a past date
  returns 0, and absent/empty/negative/NaN/infinity/invalid dates return None.
  Include a valid date without timezone and normalize it to UTC.
- [ ] Write `test_retryable_exception` for each spec exception, with write
  failures and PoolTimeout false; TLS verification ConnectError with an
  `ssl.SSLCertVerificationError` cause is false.
- [ ] Run `uv run --frozen --extra dev --extra serve pytest tests/test_gitlab_http.py -q`;
  expect failure because the new interfaces do not exist.
- [ ] Implement the interfaces using `email.utils.parsedate_to_datetime`,
  finite-number validation and the explicit policy matrix; no HTTP calls yet.
- [ ] Rerun the command; expect all Task 1 tests to pass.
- [ ] Commit: `feat(gitlab): define retry policy and Retry-After parsing`.

### Task 2: Implement shared runtime, retry executor and cooldown gates

**Files:** Modify `src/mira/platforms/gitlab/http.py`,
`tests/test_gitlab_http.py`.

**Interfaces consumed:** Task 1.

**Interfaces produced:**
- `GitLabHttpRuntime(*, policy: GitLabRetryPolicy | None = None,
  client_factory: Callable[[], httpx.AsyncClient] | None = None)`.
- `GitLabHttpRuntime.request(token: str, api_url: str, method: str, url: str,
  *, read_timeout: float = 30.0, **kwargs: Any) -> httpx.Response` (async).
- `GitLabHttpRuntime.aclose() -> None` (async, idempotent).
- `gitlab_http_scope(runtime: GitLabHttpRuntime | None = None)` returning an
  async context manager yielding `GitLabHttpRuntime`.
- `request_gitlab(token: str, api_url: str, method: str, url: str,
  *, read_timeout: float = 30.0, **kwargs: Any) -> httpx.Response` (async).

- [ ] Add MockTransport tests asserting: 503 then 200 sends twice with backoff;
  429 then 200 waits exactly the valid header; GET failures exhaust at five
  sends/four sleeps; 404 returns immediately; POST 503/ReadTimeout sends once;
  POST JSON/form 429 replays the identical body. Streaming bodies are rejected.
- [ ] Add fake-clock tests: fallback sleeps lie in `[0.5,1]`, `[1,2]`, `[2,4]`,
  `[4,8]`; 120-second header is not truncated; 301-second header records cooldown
  then fails the 300-second budget without sleeping/sending again.
- [ ] Add deterministic event-driven tests: sibling jobs share a 429 gate,
  cooldown extensions take the maximum, a final-attempt 429 still gates siblings,
  and 503 Retry-After gates siblings even when the failed request is POST.
  Assert no request semaphore permit is held during sleeps; recheck the gate
  after a queued task acquires a permit.
- [ ] Add ownership tests: concurrent first requests create one client per key;
  normalized equivalent API URLs share state; different tokens/hosts do not;
  borrowed scopes do not close clients; owned scopes close once on errors and
  cancellation; use-after-close/cross-event-loop access fail explicitly.
- [ ] Add deadline/cancellation tests for semaphore acquisition, network wait
  and cooldown; assert cancellation propagates and does not erase shared state.
  Mock clocks/sleep/random helpers inside this module, not global asyncio.
- [ ] Run the Task 1 command and confirm the new tests fail for missing runtime
  behavior before implementation.
- [ ] Implement the interfaces and exact limits from the spec. Use request-local
  auth headers, no token in logs, and no transport-level retry layer.
  Buffer normal responses; close retried responses before waiting. Start the
  monotonic deadline before queueing. Normalize and validate all outgoing API
  targets before applying credentials.
- [ ] Rerun `tests/test_gitlab_http.py`; expect all tests to pass.
- [ ] Commit: `feat(gitlab): share pooled requests and throttle cooldowns`.

### Task 3: Route every GitLab HTTP consumer through the executor

**Files:** Modify `src/mira/providers/gitlab.py`,
`src/mira/platforms/fetch.py`, `src/mira/platforms/gitlab/auth.py`,
`src/mira/platforms/gitlab/webhook.py`, `tests/test_gitlab_provider.py`,
`tests/test_repo_fetcher.py`, `tests/test_gitlab_webhooks.py`.
Add auth tests to `tests/test_gitlab_http.py`.

**Interfaces consumed:** `request_gitlab` and `gitlab_http_scope` from Task 2.
**Interfaces produced:** Existing provider/fetcher/auth signatures unchanged.

- [ ] Add provider integration tests inside one runtime scope: raw source GET,
  `_paginate` and history all recover from 503; each also recovers from 429.
  Assert pagination retries the same URL without appending duplicate entries.
- [ ] Add repository-fetcher tests covering default branch, branch SHA, tree,
  file content and archive GET through the same runtime; preserve 404/empty-repo,
  malformed JSON, cyclic links and existing keyset-query protections.
- [ ] Add identity/discovery tests for 503 recovery and shared cooldown with a
  concurrent provider request. Discovery exhaustion must raise, not return a
  successful prefix of projects.
- [ ] Add safety tests: cross-origin and outside-API-root Link targets never
  receive a request; an archive redirect does not follow with PRIVATE-TOKEN and
  returns the established fallback sentinel; logs contain neither token nor
  query/body secrets.
- [ ] Run the three existing test modules plus `tests/test_gitlab_http.py`;
  confirm the new routing assertions fail before modifying consumers.
- [ ] Replace GitLab AsyncClient usages with `request_gitlab`. Keep only one
  executor; do not introduce `_get_with_retries`. Keep `_request` as the provider's
  accepted-status adapter, but replace its current full-URL/response-body error
  text with sanitized method/origin/path/status diagnostics. Migrate GitLab branch lookup directly rather than
  changing `_fetch_branch` for other platforms. Preserve 120-second archive read
  allowance and existing archive extraction/fallback behavior.
- [ ] Update fake-client tests to use injected runtime clients/MockTransport.
  Preserve PR 3's regression cases: recovery, five-attempt exhaustion, no 404
  retry, pagination recovery, history recovery.
- [ ] Rerun the tests; expect all to pass, including non-GitLab fetcher tests.
- [ ] Commit: `refactor(gitlab): centralize provider and indexing HTTP calls`.

### Task 4: Own the runtime in server lifespan and CLI invocations

**Files:** Modify `src/mira/platforms/server.py`, `src/mira/cli.py`.
Create `tests/test_gitlab_http_lifecycle.py`; update `tests/test_gitlab_webhooks.py`
and `tests/test_cli.py` where startup assumptions change.

**Interfaces consumed:** Task 2 scopes/runtime; unchanged consumer interfaces.
**Interfaces produced:** `app.state.gitlab_http_runtime` for explicit request
middleware binding; no process-global clients or factory API changes.

- [ ] Add explicit lifespan tests that enter `app.router.lifespan_context(app)`.
  Send two overlapping GitLab jobs and one dashboard/indexing background job;
  assert all use the same runtime/client for the same URL/token.
  Do not assume HTTPX ASGITransport runs lifespan automatically.
- [ ] Add tests proving GitLab discovery is bound to that runtime, job completion
  does not close it, and shutdown cancels/awaits discovery before closing clients.
  Include cancellation-safe task completion callbacks.
- [ ] Add tests for failure inside a job and request cancellation: unrelated jobs
  retain a usable client and shared cooldown. Assert no cross-app shared state.
- [ ] Add CLI tests calling the command twice in one test process with separate
  event loops; assert separate owned runtimes, both closed, and no client created
  before the invocation's loop. Include pre-server identity bootstrap.
- [ ] Run `tests/test_gitlab_http_lifecycle.py` and relevant existing tests;
  confirm failure before implementing bindings.
- [ ] In lifespan, own one runtime and bind discovery creation inside its scope.
  In HTTP middleware, borrow `app.state.gitlab_http_runtime` for the full request
  dispatch; background tasks inherit the binding. Use explicit teardown ordering
  and await GitLab discovery cancellation before runtime closure.
- [ ] Wrap CLI review work in an async helper with an owned scope inside the
  existing `asyncio.run`; similarly scope identity bootstrap. Preserve stdin,
  local diff, dry-run and non-GitLab behavior.
- [ ] Rerun lifecycle, GitLab webhook, general webhook and CLI tests; expect pass.
- [ ] Commit: `feat(gitlab): own shared HTTP runtime in server and CLI`.

### Task 5: Surface exhausted fetches without inventing missing source

**Files:** Modify `src/mira/providers/gitlab.py`,
`src/mira/platforms/fetch.py`, `src/mira/index/context.py`,
`src/mira/core/threads.py`; update `tests/test_gitlab_provider.py`,
`tests/test_repo_fetcher.py`, `tests/test_index_context.py`,
`tests/test_engine.py`.

**Interfaces consumed:** `GitLabRequestError` from Task 1.
**Interfaces produced:** Existing success/missing-file return types unchanged;
typed transport failures remain distinguishable from legitimate absence.

- [ ] Add tests: actual file 404 retains its sentinel; exhausted source/tree GET
  raises GitLabRequestError rather than returning empty data; optional history
  returns successful entries plus an explicit warning naming failed paths.
- [ ] Add `ProviderSourceFetcher` regression: first call raises typed transient
  failure and returns None with a warning; a second call for the same path
  succeeds and is cached. Do not cache the first failure as absence.
- [ ] Add thread-verification regression with two paths: exhausted GitLab fetch
  leaves that path's threads unresolved and excludes it from LLM verification,
  while another path still verifies/resolves normally. Preserve checked/remaining
  counts and normal 404 behavior.
- [ ] Run these regression tests and confirm they fail before changing fallbacks.
- [ ] Re-raise typed failures through source/tree fallback adapters. At optional
  context boundaries warn without caching transient failure; at thread resolution
  exclude unavailable paths. Keep optional archive fallback and history best
  effort explicit. Do not add a new dashboard status or change other forges.
- [ ] Rerun provider/fetcher/context/engine tests; expect pass.
- [ ] Commit: `fix(gitlab): distinguish exhausted fetches from missing source`.

### Task 6: Prove real pooling and verify the complete change

**Files:** Create `tests/test_gitlab_http_pool.py`; update this plan's execution
checklist with verified results, not assumptions.

**Interfaces consumed:** Completed runtime and consumer integration.

- [x] Add a loopback HTTP/1.1 keep-alive server fixture with deterministic cleanup,
  correct Content-Length and a connection counter; no external service.
  Write a sequential cross-job test asserting two requests use one TCP
  connection, and a concurrent test asserting no more than eight connections.
  Disable environment proxies for the loopback client's injected factory.
  (`tests/test_gitlab_http_pool.py`: threaded `ThreadingHTTPServer`,
  `protocol_version="HTTP/1.1"`, explicit `Content-Length`, per-connection
  counter, tracked sockets closed on teardown; injected clients set
  `trust_env=False` and the production 8/8/30 limits.)
- [x] Run the new tests against the baseline/old client-per-call path and observe
  the expected reuse assertion fail; then run against the new shared runtime
  and require pass. Client identity mocks alone are not sufficient evidence.
  (RED: `MIRA_GITLAB_HTTP_POOL_LEGACY_CONTROL=1 uv run --frozen --extra dev
  pytest tests/test_gitlab_http_pool.py -q` → 1 failed:
  `assert connections == 1` / `assert 2 == 1`; GREEN default run → 3 passed.)
- [x] Run `uv run --frozen --extra dev --extra serve --extra bedrock pytest -q`.
  Include the optional Bedrock dependency so its previously observed missing-boto3
  failure does not obscure regressions. Record unrelated failures separately.
  (2924 passed, 79 skipped, 39 deselected, 21 warnings in 133.29s; no failures.
  Passed count exceeds the pre-Task-1 baseline of 2645 because Tasks 1–5 added
  tests; warning count matches the baseline's 21.)
- [x] Run Ruff check/format-check on changed Python files using
  `uv run --frozen --extra dev ruff check <changed-python-files>` and
  `uv run --frozen --extra dev ruff format --check <changed-python-files>`.
  (Both clean on `tests/test_gitlab_http_pool.py`, `tests/test_gitlab_webhooks.py`,
  `tests/test_gitlab_provider.py`, `tests/test_repo_fetcher.py`.)
- [x] Search GitLab consumers for remaining AsyncClient/direct GET calls and
  verify every GitLab path uses the executor; other platforms remain unchanged.
  Review the complete diff for duplicated retries, unsafe write replay, secret
  logging, missing cleanup and sleep-with-permit-held races.
  (GitLab provider/fetcher/auth/discovery all call `request_gitlab`; only
  `gitlab/http.py` constructs clients. Remaining `AsyncClient` calls belong to
  GitHub/Bitbucket/Forgejo/LLM/OSV/dashboard paths and are unchanged. Diff
  review found no duplicated retry loop, no automatic write replay beyond 429
  replayable bodies, no token/query/body logging, and permits are released in a
  `finally` before sleeps.)
- [x] Commit: `test(gitlab): verify shared pool and retry integration`.
- [ ] Request fresh read-only whole-branch review against the spec and tests.
  Fix verified findings with focused failing-first regressions.
- [ ] Push a new `ai/gitlab-shared-http` implementation branch and open a PR
  against fork main, linking https://github.com/netlife-pl/mira/pull/3 as
  superseded. Do not close PR 3 or merge either PR automatically.
  Do not build/restart/deploy the live service or trigger paid live reviews.

## Plan self-review

- [x] Map each design requirement to an implementation/test task.
- [x] Define runtime ownership, scoped interfaces and same-process boundaries.
- [x] Cover Retry-After, status matrix, write ambiguity, deadlines and cancellation.
- [x] Cover cross-job gates, final-attempt cooldown and real connection reuse.
- [x] Cover every discovered GitLab caller, not only review providers.
- [x] Keep deployment, distributed coordination and other forge refactors out.
- [ ] Human review of plan and execution approach before implementation.
