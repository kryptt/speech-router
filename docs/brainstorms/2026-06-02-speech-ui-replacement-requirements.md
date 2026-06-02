---
date: 2026-06-02
topic: speech-ui-replacement
---

# Replacement Speech UI + Audio-Chat (via existing Open WebUI)

## Problem Frame

Speaches is being decommissioned (its STT/TTS compute is moving to whisper.cpp on
llama-swap + standalone Kokoro-FastAPI — see Workstream 1). Speaches also provided a
Gradio Web UI: a transcription playground, a TTS playground, and voice audio-chat. We
need to preserve those capabilities for human users without keeping Speaches.

**Decision from brainstorming:** rather than build a bespoke app, **reuse the cluster's
existing Open WebUI** (`webui` deployment, `llms.hr-home.xyz`), which already provides
STT/TTS playground settings and a voice "Call" mode against OpenAI-compatible backends,
already has Zitadel OIDC SSO (role-gated to `member`), and is scale-to-zero via Sablier.
The work is **configuration + a runbook**, not a build.

## Requirements

**Audio configuration (Open WebUI)**
- R1. Configure Open WebUI's **STT** engine (OpenAI-compatible) to call speech-router's
  `/v1/audio/transcriptions`, so users can upload/record audio → text.
- R2. Configure Open WebUI's **TTS** engine (OpenAI-compatible) to call speech-router's
  `/v1/audio/speech` with a Kokoro voice, so users can do text → speech with voice
  selection.
- R3. The voice **Call mode** works end-to-end as audio-chat: record speech → STT
  (R1) → LLM (existing ollama-router connection) → TTS (R2) → playback. Two known
  Open WebUI gotchas must be verified, because they silently bypass the backend even
  when admin config is correct: (a) the **per-user** "Speech-to-Text Engine" must be
  **Default**, not "Web API" (Web API uses the browser and never calls speech-router;
  also browser-dependent, e.g. weak in Firefox); (b) the per-user "Display Emoji in
  Call" setting has broken Call-mode read-aloud in some versions — verify TTS plays.
- R4. Configuration is done via the Open WebUI **admin Settings UI** (persisted to the
  data volume), consistent with how the existing LLM connection is managed. (Open WebUI
  also supports `AUDIO_*` env vars, which would be GitOps-declarative — deliberately
  *not* chosen, to match the existing LLM-connection management; the runbook in R5
  offsets the reproducibility cost.)
- R7. speech-router's `/v1/audio/speech` (and `/v1/audio/transcriptions`) must
  **tolerate/ignore unexpected request fields** that Open WebUI sends (e.g. `chat_id`),
  rather than 400-ing — a documented Call-mode failure against custom backends.
  (Cross-cuts Workstream 1 — speech-router is the audio front door.)

**Reproducibility**
- R5. Capture the exact audio settings in a **runbook** in-repo so the config can be
  re-applied if the persisted data volume is lost. Record: the **Open WebUI image
  tag** in use; STT — `AUDIO_STT_ENGINE=openai`, `AUDIO_STT_OPENAI_API_BASE_URL`
  (→ speech-router), `AUDIO_STT_OPENAI_API_KEY` (placeholder), `AUDIO_STT_MODEL`
  (whatever model id speech-router/whisper accepts); TTS — `AUDIO_TTS_ENGINE=openai`,
  `AUDIO_TTS_OPENAI_API_BASE_URL`, `AUDIO_TTS_OPENAI_API_KEY` (placeholder),
  `AUDIO_TTS_MODEL` (e.g. `kokoro`), `AUDIO_TTS_VOICE` (Kokoro voice string, e.g.
  `af_heart`). **STT and TTS base URLs are independent keys** — don't assume one shared
  URL. Field names/format are version-sensitive; capture them against the pinned tag.

**Decommission**
- R6. Remove the `speaches.hr-home.xyz` IngressRoute when Speaches is removed; the UI
  lives solely at `llms.hr-home.xyz`. (Executed in Workstream 1's phase-out cleanup.)

## Success Criteria

- A `member`-role user at `llms.hr-home.xyz` can transcribe audio and synthesize
  Kokoro TTS — all served by speech-router/Kokoro/ollama-router with **no Speaches
  dependency**.
- Voice Call-mode audio-chat works end-to-end **once the R3 gotchas are verified**
  (per-user STT engine = Default; TTS plays in Call mode); confirm by observing a
  request actually hit speech-router's `/v1/audio/transcriptions` during a Call.
- `speaches.hr-home.xyz` no longer resolves to a service.
- The audio settings are documented well enough to re-create from scratch.

## Scope Boundaries

- **Not** building a bespoke web app (reuse Open WebUI).
- **Not** changing Home Assistant's voice path — HA uses speech-router's Wyoming TCP
  server directly, independent of this UI.
- **Not** changing Open WebUI's existing LLM chat connection to ollama-router (already
  working via the admin DB).
- **Not** moving Open WebUI off `llms.hr-home.xyz` or changing its OIDC/Sablier setup.

## Key Decisions

- **Reuse Open WebUI, don't build.** It already covers the Speaches UI feature set
  (STT/TTS playground + voice Call) and is already OIDC-secured and deployed. Near-zero
  carrying cost vs. a bespoke app.
- **Configure via admin UI (persisted), with a runbook for reproducibility.** Matches
  how the LLM connection is already managed; the runbook (R5) offsets the loss of
  GitOps-declarativeness.
- **speech-router needs no auth layer of its own for this.** Access control is
  *delegated*: OIDC at the Open WebUI ingress authenticates the user, and in-cluster
  NetworkPolicy restricts who can reach speech-router (accepted homelab posture from
  Workstream 1). Open WebUI's OpenAI audio config uses a **non-empty placeholder** API
  key (e.g. `not-needed`) for both STT and TTS — an empty key can behave differently in
  some code paths.

## Dependencies / Assumptions

- **Depends on Workstream 1**: speech-router must expose working OpenAI-style
  `/v1/audio/transcriptions` and `/v1/audio/speech` (Kokoro) endpoints — **and tolerate
  unexpected fields per R7** — before this config is meaningful. WS2 config happens
  after WS1's data path is live.
- **Research gate for R1–R3:** the audio-settings field names, custom-base-URL support,
  Call-mode engine routing, and Kokoro voice format must be confirmed **against the
  pinned Open WebUI image tag** before implementing R1–R3 (Open WebUI audio settings
  and Call-mode behavior have changed across releases). Pin/record the tag.

## Outstanding Questions

### Deferred to Planning

- [Affects R1,R2][Needs research] The `AUDIO_*` field names are captured in R5 from
  current docs, but **confirm they match the deployed image tag** (settings have
  shifted across releases) and confirm the exact Kokoro voice string Open WebUI's TTS
  expects.
- [Affects R3][Needs research] Confirm Open WebUI's Call mode honors the configured
  OpenAI STT/TTS engines (vs. only browser/local engines).
- [Affects success criteria][Optional] Open WebUI currently requests `nvidia.com/gpu: 1`
  on hr-main; with STT/TTS/LLM all remote, evaluate dropping the local GPU request to
  reduce hr-main GPU pressure (aligns with the overall goal; Open WebUI is scale-to-zero
  so impact is intermittent).

## Next Steps

→ `/ce:plan` for structured implementation planning (after, or alongside, Workstream 1).
