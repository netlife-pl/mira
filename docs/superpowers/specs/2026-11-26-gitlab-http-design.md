# Shared GitLab HTTP transport

## Intent and scope

Replace the partial retry implementation proposed in
https://github.com/netlife-pl/mira/pull/3 with a clean, shared GitLab HTTP
transport. Concurrent review, indexing, discovery and identity requests using
the same API URL/token must reuse connections and cooperate when throttled.

This document records the agreed direction. Numerical defaults and the scoped
runtime binding below are engineering choices for implementation, not previously
specified user requirements.

## Findings

- PR 3 duplicates retry loops; pagination/history omit GET 5xx retries.
- `_request` creates a client on every attempt.
- Its delay parser ignores HTTP dates and shortens server waits to 60 seconds.
- `GitLabRepoFetcher`, discovery and `GitLabTokenAuth` use independent clients.
- FastAPI already has a lifespan owner; the CLI uses separate `asyncio.run`
  invocations, so a module-global async client would be unsafe.
- Optional source/history callers can swallow errors. Retrying is not evidence
  that every source fetch succeeded.

## Approaches

1. Patch PR 3's individual loops: smallest diff, but leaves policy drift and
   independent connection pools. Rejected.
2. One provider/job-owned client: reuses some connections but does not coordinate
   jobs sharing a token. Rejected following the user's ownership correction.
3. Application-owned shared transport with invocation-owned CLI scope:
   recommended and selected direction. One retry executor, one pool and one
   throttle gate per normalized API URL/token within that runtime.

## Ownership and interfaces

Create `src/mira/platforms/gitlab/http.py`. `GitLabHttpRuntime` owns lazy clients,
semaphores and monotonic cooldowns, keyed by normalized API URL and a non-logged
token digest. With the deployment's single token there is one shared pool.
Different credentials must not share cookies or throttle state.

Use `gitlab_http_scope(runtime=None)` as an async context manager. Without an
argument it owns a runtime; with an argument it only binds/borrows it through a
ContextVar. `request_gitlab(...)` resolves that runtime at request time, so
provider/fetcher factories and `BaseProvider` need no new lifecycle API.

FastAPI lifespan owns the runtime. Explicitly bind it in HTTP middleware:
lifespan ContextVars alone do not propagate into ASGI request tasks. Background
tasks inherit the request binding; lifespan discovery tasks have an explicit
binding. Shutdown stops/awaits owned GitLab discovery before closing clients.
The server must drain request background work before lifespan teardown.

CLI review and pre-server identity lookup each bind an owned scope inside their
own `asyncio.run`. Direct library calls outside a scope use a temporary runtime
for that logical request and the same retry policy, not a leaked global client.
Library users wanting cross-call pooling use the scope explicitly.

## Request policy

- Five total attempts per logical HTTP request.
- Overall deadline: 300 seconds, covering queueing, attempts and sleeps.
- Pool and active-request limits: eight each per API URL/token; eight keep-alive
  connections, 30-second keep-alive expiry.
- HTTPX connect/write/pool timeouts: 10/30/5 seconds; read timeout: 30 seconds.
  Preserve the archive caller's 120-second read timeout within the same deadline.
- GET/HEAD retry statuses: 408, 429, 500, 502, 503, 504. Other statuses are not
  blanket-retried, including 401/403/404/409/422/501/505 and redirects.
- Other methods retry only an explicit 429 with a replayable payload.
  No automatic POST/PUT/PATCH/DELETE replay on 5xx or transport ambiguity.
- GET/HEAD retry `ConnectError`, `ConnectTimeout`, `ReadError`, `ReadTimeout`,
  `WriteError`, `WriteTimeout` and `RemoteProtocolError`.
  `PoolTimeout`, invalid URLs, TLS certificate verification failures and
  cancellation are not transient remote failures.
- Bodies must be replayable bytes/text/JSON/form data. Do not accept streaming
  request bodies for automatic retry. Existing GitLab writes use replayable data.
- Do not enable a second layer of HTTPX transport retries.

Honor `Retry-After` on retryable responses: non-negative delay-seconds or
HTTP-date relative to UTC. Accept finite fractional seconds for compatibility;
negative, malformed and non-finite numbers fall back to backoff. Past dates mean
zero delay. Convert the result to a monotonic not-before instant once.

No server-directed delay is capped downward. If it cannot fit the request's
remaining deadline, record any shared cooldown first, then raise an explicit
error without another send. Never sleep after the final attempt.

Without a usable header, use equal-jitter exponential backoff:
`cap = min(60, 2 ** (attempt - 1))`, delay uniformly in `[cap / 2, cap]` seconds.
Add no negative jitter to a server-specified wait.

## Shared throttle gate

Every 429 extends a shared cooldown, even on the last attempt; never shorten an
existing cooldown. If there is no usable header, use that attempt's backoff.
503 with a valid `Retry-After` also extends the shared gate, including for a
non-replayed write, to respect explicit service-wide unavailability guidance.
Other retryable responses wait only within their logical request.

Before dispatch: wait for the gate, acquire the request semaphore, recheck the
gate, and release/restart if another task extended it. Do not sleep with a
semaphore permit or response connection held. In-flight requests may complete;
this is coordination, not a proactive requests-per-second scheduler.

Enforce the deadline around queueing, network work and sleeps with Python 3.11
asyncio timeout support. Check budgets explicitly before sleeping/sending.
Cancellation must release permits but must not reset another job's cooldown.

## Integration and failure semantics

All GitLab HTTP calls in providers, repository fetching, discovery and identity
use `request_gitlab`. Keep response interpretation with callers: accepted 404
and empty repositories must retain their existing meaning; pagination resumes
the failed page and never returns a successfully fetched prefix as a full tree.
Preserve existing tree validation and JSON validation.

The executor returns non-retried responses and successful responses. It raises
`GitLabRequestError(ProviderError)` for exhausted retries, overall deadline and
terminal transport failures, carrying reason, attempt count and optional status.
Logs contain method, origin/path, status/reason, attempt and delay; never tokens,
query strings, request bodies or full response bodies.

Provider source/tree fetches propagate this typed failure rather than converting
it to empty content. Optional context may still continue, but records a warning
and does not cache a transient failure as a missing file. Thread verification
must leave affected threads unresolved instead of submitting fabricated empty
source to the LLM. Optional history may return partial results with a warning
identifying failed paths. No new dashboard/metrics schema in this change.

Reject authenticated pagination/redirect targets outside the configured API
origin/root. Preserve tree's stricter existing endpoint/query checks. Do not
automatically follow redirects with `PRIVATE-TOKEN`; archive redirect responses
use the existing per-file fallback rather than introduce cross-origin downloads.

## Non-goals

No cluster support, Redis, distributed locks, new dependencies, generic transport
framework for other forges, LLM changes, archive strategy changes, persistent
request caching, automatic write reconciliation, deployment or Ansible edits.

## Verification

Fake clocks/sleep/randomness and HTTPX mock transports prove retry/gate policy.
A small loopback HTTP/1.1 server proves real keep-alive reuse (mock transports
cannot prove it). ASGI lifespan tests prove shared ownership, including dashboard
indexing jobs, and repeated CLI invocations prove event-loop isolation.
Tests must not require GitLab credentials, LLM calls or real retry sleeps.
