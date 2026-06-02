---
title: "feat: Replacement speech UI via Open WebUI audio config (Workstream 2)"
type: feat
status: active
date: 2026-06-02
origin: docs/brainstorms/2026-06-02-speech-ui-replacement-requirements.md
---

# feat: Replacement speech UI via Open WebUI audio config (Workstream 2)

## Overview

Speaches' Gradio UI (transcription playground, TTS playground, voice audio-chat) is
being decommissioned with Speaches (Workstream 1). Rather than build a bespoke app, we
**reuse the cluster's existing Open WebUI** (`llms.hr-home.xyz`, already Zitadel-OIDC +
Sablier scale-to-zero) by configuring its STT and TTS engines to call speech-router, and
using its voice **Call mode** as audio-chat. This is a **configuration + runbook** task —
no application code, no new service. The final step coordinates dropping the
`speaches.hr-home.xyz` IngressRoute (owned by WS1's phase-out).

## Problem Frame

Human users need the Speaches-UI capabilities (transcribe, synthesize, voice chat)
without Speaches. Open WebUI already provides all three against OpenAI-compatible
backends and is already deployed, authenticated, and on-demand — so the gap is purely
its **audio configuration** plus a reproducibility runbook (since that config is
persisted in Open WebUI's data volume, not GitOps-declarative).

## Requirements Trace

- **R1.** Open WebUI STT engine (OpenAI-compatible) calls speech-router
  `/v1/audio/transcriptions` (upload/record → text). *(origin R1)*
- **R2.** Open WebUI TTS engine (OpenAI-compatible) calls speech-router
  `/v1/audio/speech` with a Kokoro voice (text → speech). *(origin R2)*
- **R3.** Voice **Call mode** works end-to-end as audio-chat: speech → STT → LLM (existing
  ollama-router connection) → TTS → playback, with the per-user gotchas verified. *(origin R3)*
- **R4.** Audio is configured via the Open WebUI **admin Settings UI** (persisted to the
  data volume). *(origin R4)*
- **R5.** A **runbook** in-repo captures the image digest + exact audio settings so the
  config can be re-applied if the data volume is lost. *(origin R5)*
- **R6.** The `speaches.hr-home.xyz` IngressRoute is removed once Open WebUI audio is
  confirmed (coordinated with WS1's phase-out). *(origin R6)*

## Scope Boundaries

- **Non-goal:** building a bespoke UI (reuse Open WebUI).
- **Non-goal:** changing Home Assistant's voice path (Wyoming via speech-router, separate).
- **Non-goal:** changing Open WebUI's existing LLM/ollama-router connection, OIDC, or
  Sablier setup.
- **Non-goal:** speech-router code changes — R7 (tolerate extra fields like `chat_id`) is
  owned by WS1 and is a **dependency** here, not work in this plan.
- **Non-goal:** moving audio config to declarative `AUDIO_*` env vars (deliberately using
  the admin UI to match how the LLM connection is managed; the runbook offsets it).

## Context & Research

### Relevant Code and Patterns

- hr-fleet `fleet/ai/webui.yaml` — the Open WebUI Deployment. Already: Zitadel OIDC
  (`OAUTH_*`, role-gated to `member`), Sablier scale-to-zero middleware, host
  `llms.hr-home.xyz`, image pinned `ghcr.io/open-webui/open-webui:cuda@sha256:19d3…`,
  data on hostPath `/swarm/main/ai/open-webui`. **No** `OPENAI_API_BASE_URL`/`OLLAMA_*`
  env → connections live in the persisted admin DB (confirming the admin-UI approach).
- hr-fleet `fleet/ai/speech-router.yaml` — the `speaches.hr-home.xyz` IngressRoute to
  remove (WS1 phase-out).
- WS1 plan `docs/plans/2026-06-02-001-…` — provides the STT/TTS endpoints this config
  targets and the R7 field-tolerance.

### Institutional Learnings

- From the brainstorm's feasibility review (cited there): Open WebUI supports independent
  `AUDIO_STT_OPENAI_API_BASE_URL` and `AUDIO_TTS_OPENAI_API_BASE_URL` (separate keys);
  Kokoro voice strings like `af_heart`; default key `not-needed` (must be **non-empty**);
  Call mode honors the configured server engines **only if** the per-user STT engine is
  "Default" (not "Web API"); "Display Emoji in Call" has broken Call-mode TTS in some
  versions; audio-settings field names have shifted across releases (confirm vs the
  pinned tag).

### External References

- Open WebUI audio docs (STT/TTS env + admin settings); Kokoro-FastAPI integration guide
  (model `kokoro`, voices, key `not-needed`) — see origin doc's Sources.

## Key Technical Decisions

- **Reuse Open WebUI, configure via admin UI + runbook** *(origin)* — near-zero carrying
  cost; the runbook (R5) offsets non-declarative config.
- **Non-empty placeholder API keys** for STT and TTS — empty keys behave differently in
  some code paths; speech-router ignores auth (accepted LAN posture).
- **Drop `speaches.hr-home.xyz` only after** Open WebUI audio is confirmed — avoids a UI
  gap (the WS1 phase-out is gated on this confirmation).

## Open Questions

### Resolved During Planning

- Reuse vs build → reuse Open WebUI. Config method → admin UI + runbook. Old host → drop
  `speaches.hr-home.xyz`, UI lives at `llms.hr-home.xyz`. Auth → existing OIDC + LAN posture.

### Deferred to Implementation

- **[Needs research]** Exact audio-settings field labels/values for the deployed Open
  WebUI image (`@sha256:19d3…`) — confirm STT/TTS each take a custom base URL + model +
  voice, and the precise Kokoro voice string the TTS engine expects. Field names have
  shifted across releases.
- **[Needs verification]** That Call mode uses the configured server STT/TTS (not browser
  Web Speech) for a `member` user, and that TTS plays in Call mode on this version.
- **[Optional]** Open WebUI currently requests `nvidia.com/gpu: 1`; with STT/TTS/LLM all
  remote, evaluate dropping the local GPU request to reduce hr-main pressure (scale-to-zero
  limits impact; out of scope for this plan unless trivial).

## Implementation Units

- [ ] **Unit 1: Configure Open WebUI STT + TTS engines (admin Settings)**

**Goal:** Point Open WebUI's audio engines at speech-router so transcription and TTS work.

**Requirements:** R1, R2, R4

**Dependencies:** WS1 endpoints live (`/v1/audio/transcriptions`, `/v1/audio/speech`
serving Kokoro, tolerating extra fields per WS1 R7).

**Files:** None in-repo (config persisted to Open WebUI's data volume); settings captured
in Unit 3's runbook.

**Approach:**
- In Admin → Settings → Audio: set **STT** engine = OpenAI, base URL → speech-router's
  in-cluster Service for transcriptions, model = the id speech-router/whisper accepts,
  non-empty placeholder key. Set **TTS** engine = OpenAI, its **own** base URL →
  speech-router speech endpoint, model `kokoro`, voice e.g. `af_heart`, non-empty
  placeholder key. (STT and TTS base URLs are independent.)
- First confirm the exact field labels against the deployed image (Deferred [Needs
  research]); the field names are version-sensitive.

**Test scenarios:** Test expectation: none — UI configuration. Validation: a transcription
request from Open WebUI returns text from speech-router; a TTS request returns Kokoro
audio (observe the calls hit speech-router).

**Verification:** transcription playground returns text; TTS playground plays a Kokoro
voice; both observed hitting speech-router (not OpenAI/browser).

- [ ] **Unit 2: Verify Call-mode audio-chat end-to-end + per-user settings**

**Goal:** Confirm the voice Call mode works as audio-chat and document the per-user
settings that silently bypass the backend.

**Requirements:** R3

**Dependencies:** Unit 1; existing ollama-router LLM connection.

**Files:** None (verification); findings recorded in Unit 3's runbook.

**Approach:**
- As a `member` user, ensure personal **Speech-to-Text Engine = Default** (not "Web API",
  which uses the browser and never calls speech-router). If Call-mode TTS doesn't play,
  toggle **"Display Emoji in Call" off** (known bug on some versions).
- Run a Call: speak → confirm a request hits speech-router `/v1/audio/transcriptions` →
  LLM replies via ollama-router → TTS plays via speech-router.

**Test scenarios:** Test expectation: none — operational verification. Acceptance: a full
voice turn completes and a transcription request is observed reaching speech-router during
the Call.

**Verification:** end-to-end voice turn works; the required per-user settings are
identified and recorded.

- [ ] **Unit 3: Reproducibility runbook**

**Goal:** Capture the config so it can be re-created if the data volume is lost.

**Requirements:** R5

**Dependencies:** Units 1–2 (records their confirmed values).

**Files:** Create an in-repo runbook (e.g. `docs/runbooks/open-webui-speech-config.md`).

**Approach:**
- Record: the Open WebUI **image digest** in use; the exact STT settings (engine, base
  URL, model, placeholder key); TTS settings (engine, base URL, model `kokoro`, voice,
  placeholder key); the per-user gotchas from Unit 2 (STT engine = Default; Display Emoji
  in Call off). Note these are version-sensitive to the recorded digest.

**Test scenarios:** Test expectation: none — documentation.

**Verification:** a reader can re-apply the audio config from the runbook against the
recorded image digest without rediscovery.

- [ ] **Unit 4: Cutover coordination — confirm replacement, trigger Speaches ingress removal**

**Goal:** Once Open WebUI is the working replacement, signal WS1 to drop
`speaches.hr-home.xyz`.

**Requirements:** R6

**Dependencies:** Units 1–3.

**Files:** Modify (WS1-owned, coordinated) hr-fleet `fleet/ai/speech-router.yaml` —
remove the `speaches.hr-home.xyz` IngressRoute as part of WS1's dated phase-out.

**Approach:**
- Confirm transcription, TTS, and Call mode all work via Open WebUI (Units 1–2) and the
  runbook exists (Unit 3). Then trigger WS1's phase-out removal of the Speaches
  IngressRoute (and the rest of the Speaches sidecar per WS1 Unit 7).

**Test scenarios:** Test expectation: none — coordination/decommission.

**Verification:** `speaches.hr-home.xyz` no longer resolves to a service; Open WebUI at
`llms.hr-home.xyz` covers all three capabilities with no Speaches dependency.

## System-Wide Impact

- **Interaction graph:** Open WebUI → speech-router (STT/TTS) + ollama-router (LLM, existing).
  No change to HA's Wyoming path or to speech-router code.
- **Error propagation:** if speech-router STT is down, Open WebUI surfaces its own error;
  unaffected by this config work.
- **Unchanged invariants:** Open WebUI OIDC, Sablier, host, and LLM connection; HA voice
  via Wyoming.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Audio-setting field names differ on the deployed Open WebUI version | Confirm against the pinned digest before configuring (Unit 1 / Deferred research). |
| Call mode bypasses speech-router (per-user "Web API" STT) or TTS silent (Emoji-in-Call bug) | Unit 2 verifies + documents the required per-user settings. |
| Config lost with the data volume (non-declarative) | Unit 3 runbook captures it against the image digest. |
| Dropping `speaches.hr-home.xyz` before the replacement works → UI gap | Unit 4 gates removal on Units 1–3 confirmation. |
| **Hard dependency on WS1** (endpoints + R7 field tolerance) | Sequence WS2 after WS1 Units 1–5 are live; verify R7 (no 400 on `chat_id`) during Unit 2. |

## Documentation / Operational Notes

- The runbook (Unit 3) is the durable artifact. Coordinate Unit 4 with WS1's dated
  Speaches removal. Consider (optional, separate) dropping Open WebUI's `nvidia.com/gpu`
  request now that audio/LLM are remote.

## Sources & References

- **Origin document:** `docs/brainstorms/2026-06-02-speech-ui-replacement-requirements.md`
- Related plan: `docs/plans/2026-06-02-001-feat-stt-offload-strix-halo-plan.md` (WS1)
- Manifests: hr-fleet `fleet/ai/webui.yaml`, `fleet/ai/speech-router.yaml`
- External: Open WebUI audio docs; Kokoro-FastAPI integration (see origin Sources).
