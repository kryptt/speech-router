# ce:review — Unit 3 ordered STT failover (mode:autofix)

- Scope: working tree vs `main` (77e208a). Files: Cargo.toml, src/{asr,main,metrics,proxy,wyoming}.rs.
- Plan: docs/plans/2026-06-02-003-feat-cuda-node-stt-failover-plan.md (Unit 3).
- Reviewers (9): correctness, reliability, adversarial, performance, testing, maintainability,
  project-standards, agent-native, learnings-researcher.
- Verdict: **Ready** (after fixes below).

## Findings fixed

| # | Sev | Finding | Reviewers | Fix |
|---|-----|---------|-----------|-----|
| 1 | P1 | `/health` probed only the primary upstream → a primary outage fails readiness and pulls speech-router from Service endpoints, defeating CUDA failover | reliability, agent-native | `health_route` now probes all upstreams, healthy if ANY responds |
| 2 | P2 | Body-read transport error after a successful `send()` returned 502 without failover (buffered asr/wyoming paths) | adversarial, reliability, correctness | read body inside the match; on transport failure (non-last) fail over; `served` recorded only after a full body |
| 3 | P2 | Last-upstream transport failure not recorded in metric | correctness | record `fell_through` for the final upstream's transport failure (all 3 sites) |
| 4 | P1(test) | No test for `main.rs::passthrough_transcriptions` failover (Open WebUI path) | testing, maintainability | added 3 bin tests: fails-over, all-down→502, 503-not-failed-over |
| 5 | P2/P3(test) | No Wyoming all-down test; loose `>100` body assertions | testing | added wyoming all-down→Err test; tightened assertions to `>=2048`/`>=3244` |

## Rejected (false positives / out of scope)

- **Metric label cardinality "explosion"** (adversarial P2): labels come from static `STT_URLS` config
  (bounded 2×N), not pod IPs/service-discovery — performance reviewer rebutted. No change.
- **TLS errors absent from `is_transport_failure`** (reliability P2): upstreams are in-cluster `http://`,
  no TLS handshake occurs. Out of scope.
- **3-way duplication + `too_many_arguments`** (maintainability P2/P3): the three call sites have
  genuinely different body construction (reopen file / clone wav / clone Bytes) and return types
  (Response vs Event); reviewer acknowledged the duplication is justified. Avoided over-abstraction.
- **primary_base rename, translate-task failover test**: low value; failover loop is task-agnostic.

## Known limitation (residual risk, documented)

- A clean mid-body connection RESET on the primary that reqwest classifies as a body error (not
  is_connect/is_timeout) surfaces as 502 rather than failing over. The predicate is deliberately
  connect/timeout-only (the design forgoes a request timeout for long Bazarr transcriptions). The
  dominant outage mode (node down → connection refused) IS handled and tested.

## Validation

- 65 tests pass (62 lib + 3 bin) single-threaded; fmt clean; clippy introduces no new warnings
  (6 pre-existing `collapsible_if` in untouched asr/ffmpeg/wyoming code remain).
- Build/test run in a trixie container on rh-anine (host can't build: FFmpeg 8 vs ffmpeg-sys-next).
- Pre-existing: 2 config.rs env-var tests race under parallel execution (pass with --test-threads=1).
