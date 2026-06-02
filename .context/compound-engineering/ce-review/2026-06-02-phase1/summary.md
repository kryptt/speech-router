# ce:review (autofix) — Phase 1 (Units 1–3), branch feat/stt-llama-swap-phase1

Scope: uncommitted working tree vs main(ff62516) — src/{config,asr,wyoming,main,metrics}.rs (+589/−135).
Reviewers: correctness, testing, maintainability, reliability, api-contract, adversarial (haiku).
Tests: 54 passed / 0 failed (trixie container on rh-anine). fmt clean.

## Applied (safe_auto)
- Reliability — added `connect_timeout(10s)` on the shared reqwest client (fast-fail on dead/unreachable backend; NOT a total timeout, so long Bazarr transcriptions are unaffected). main.rs.
- Reliability — `health_route` STT probe now has a 5s total timeout. main.rs.
- Reliability — Wyoming TTS request now has a 30s total timeout. wyoming.rs.
- Maintainability — renamed stale operator-facing "speaches …" error strings to "stt backend"/"tts backend" in asr.rs + wyoming.rs.

## Resolved after review
- **[P1] `/v1/audio/translations` via OpenAI passthrough silently transcribed** — FIXED: passthrough now returns `501 Not Implemented` with a message pointing to `/asr` (translate task), instead of silently transcribing. Failing loud beats failing wrong; full OpenAI-passthrough translation (buffered multipart rewrite) deferred (no consumer). 54 tests still green.

## Residual actionable (deferred follow-up)
- **[P2] Detection quorum tie / low-vote weak result** — with 13 chunks, quorum=2; if only 2 succeed and disagree ({en:1, es:1}), `max_by_key` picks arbitrarily (nondeterministic). Consider requiring a plurality margin or a higher effective quorum. (asr.rs detect_language_from_file)
- **[P2] Testing gaps for the new async/HTTP paths** — passthrough `upstream_for_path` routing, `/v1/models` synthesis, `health_route` STT-probe, quorum-failure branch, semaphore bounding, `translate=true` form submission, empty-transcript metric increment. Core pure-logic (config parsing, `normalize_language_code`, `transcription_url`) IS unit-tested; the gaps are integration-test territory (need a mock HTTP backend).
- **[P2] Extract the duplicated `stt_upstreams.first()` selection** (appears in handle_asr, handle_detect_language, passthrough_route, finish_stt) into a shared helper, and extract `upstream_for_path` into a testable pure fn. Improves consistency + enables routing unit tests.

## Accepted / not bugs
- Main transcription total timeout intentionally unbounded (full-movie Bazarr jobs) — `connect_timeout` covers the dead-backend case.
- Env-var migration (STT_URLS/TTS_URL/STT_MODEL required) is coordinated in WS1 Unit 7 cutover; config errors loudly if unset (tested).
- `DEFAULT_MODEL` honored as `STT_MODEL` fallback (works as intended).
- Config/AsrState struct changes are internal (binary crate; no external library consumers).
- Wyoming advertised `name:"speaches"` / program labels left unchanged (contract surface for HA) except the model description ("whisper.cpp …").

## Verdict
Ready with fixes. Safe reliability + clarity fixes applied and verified (54 tests green). The P1 translations gap has no consumer in this deployment but should get an explicit decision before the OpenAI passthrough is advertised broadly. Remaining items are P2 hardening/tests suitable for a follow-up.
