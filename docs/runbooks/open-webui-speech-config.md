# Runbook: Open WebUI speech (STT/TTS) configuration

Configures the existing Open WebUI (`llms.hr-home.xyz`, ai namespace) to use
speech-router as its STT/TTS backend — the WS2 replacement for the Speaches
Gradio UI. speech-router is a **transparent OpenAI-compatible audio backend**, so
Open WebUI just points its "OpenAI" STT/TTS engines at it; it needs no knowledge
of the whisper.cpp / Kokoro backends behind it.

This config lives in Open WebUI's **persisted DB** (hostPath
`/swarm/main/ai/open-webui`), **not** in env vars — `AUDIO_*` env vars are ignored
on an already-running instance (`ENABLE_PERSISTENT_CONFIG` defaults true). Re-apply
these steps if that volume is ever lost.

- Open WebUI image (pinned digest, for reproducibility):
  `ghcr.io/open-webui/open-webui:cuda@sha256:19d3ec8e516e49bea271e5b939c2ff9aefc2e7bd5b3deea248084a7e76d6eb37`

## Prerequisites (all satisfied)

- speech-router `0.4.0` deployed, routing STT→whisper / TTS→Kokoro (WS1 cutover).
- whisper + whisper-translate (rh-anine llama-swap) and Kokoro-FastAPI deployed + validated.
- NetworkPolicy **`speech-router-ingress`** admits `app=webui` → `speech-router:8000`
  (in `fleet/ai/speech-router.yaml`). Without it, default-deny blocks Open WebUI →
  speech-router.

## Steps — Admin Panel → Settings → Audio

> Base URL must include the `/v1` suffix (Open WebUI appends `/audio/transcriptions`
> and `/audio/speech`). The API key just needs to be non-empty (speech-router ignores it).

**Speech-to-Text (STT):**
- Engine: **OpenAI**
- API Base URL: `http://speech-router.ai.svc:8000/v1`
- API Key: `not-needed`
- STT Model: `whisper`
  *(speech-router routes STT by its own `STT_MODEL`, ignoring this field — any value works; `whisper` for clarity.)*
- **Save**

**Text-to-Speech (TTS):**
- Engine: **OpenAI**
- API Base URL: `http://speech-router.ai.svc:8000/v1`
- API Key: `not-needed`
- TTS Model: `kokoro`  *(required — forwarded to Kokoro-FastAPI, which validates it)*
- TTS Voice: `af_heart`  *(required — must be a real Kokoro voice; others: `am_michael`, `bf_emma`, …)*
- **Save**

## Per-user settings (each user who uses voice) — Settings → Audio / Interface

These are **per-user** and silently bypass the server backend if wrong:
- **Speech-to-Text Engine: `Default`** (NOT "Web API"). "Web API" uses the browser's
  recognition and never calls speech-router.
- **"Display Emoji in Call": OFF** (Interface). When ON it breaks Call-mode TTS playback.

## Verification

1. **Transcription:** in a chat, record or upload audio → text appears.
2. **TTS:** click read-aloud on a message → hear the Kokoro `af_heart` voice.
3. **Call mode (audio-chat):** start a voice call → speak → reply is transcribed,
   answered by the LLM (ollama-router), and read back via TTS.
4. **Confirm the path:** `kubectl -n ai logs deploy/speech-router -c speech-router`
   shows the requests; `kubectl -n ai logs deploy/llama-swap` shows whisper serving.

## Alternative: apply via the admin API (scriptable, no UI)

Open WebUI exposes an admin-only config API (writes the same DB, survives restarts):
- `GET /api/v1/audio/config` (admin bearer token) → current config.
- `POST /api/v1/audio/config/update` with `{ "stt": {...}, "tts": {...} }` — round-trip
  the GET result, setting `stt.ENGINE="openai"`, `stt.OPENAI_API_BASE_URL=".../v1"`,
  `stt.OPENAI_API_KEY="not-needed"`, `stt.MODEL="whisper"`; and `tts.ENGINE="openai"`,
  `tts.OPENAI_API_BASE_URL=".../v1"`, `tts.OPENAI_API_KEY="not-needed"`,
  `tts.MODEL="kokoro"`, `tts.VOICE="af_heart"`. All sub-form fields are required, so
  GET-mutate-POST rather than sending a partial object.

## After this works

Coordinate WS1's dated **full Speaches removal**: drop the `speaches.hr-home.xyz`
IngressRoute and the Speaches sidecar from `fleet/ai/speech-router.yaml` (the
GPU-less sidecar is only kept until this UI replacement is confirmed).
