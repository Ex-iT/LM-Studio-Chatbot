# LM TTS (LM Studio + Kokoro)

Small Flask app that lets you chat with an LM Studio model and hear the replies via Kokoro text-to-speech. The `web/` frontend talks to two endpoints: `/api/chat` for combined text + audio replies and `/api/tts` for on-demand speech.

## Requirements

- Python 3.10+ and below 3.14 with `pip`
- LM Studio running locally with a chat model loaded and the API server enabled (defaults to `http://127.0.0.1:1234/v1` and key `lm-studio`)

## Quickstart

- **Windows (simplest):** double-click `run_server.bat` — it will create/activate `.venv`, install requirements, and start the server.
- **Mac/Linux (or manual):**

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

  # Start LM Studio separately, then run the app:
  python server.py
  ```

  Then open http://localhost:5000 in your browser to use the chat UI.

## Config (Arguments & Environment Variables)

All settings can be passed as command-line arguments or environment variables. Arguments take precedence.

| Argument | Env Variable | Description | Default |
| :--- | :--- | :--- | :--- |
| `--base-url` | `LMSTUDIO_BASE_URL` | Base URL for the LM Studio API | `http://127.0.0.1:1234/v1` |
| `--api-key` | `LMSTUDIO_API_KEY` | API key for LM Studio | `lm-studio` |
| `--model-name` | `LMSTUDIO_MODEL_NAME` | Force a specific model ID | (auto-detect) |
| `--repo-id` | `KOKORO_REPO_ID` | HF repo for Kokoro weights | `hexgrad/Kokoro-82M` |
| `--lang` | `KOKORO_LANG` | Default language code | `a` |
| `--voice` | `KOKORO_VOICE` | Default voice name | `af_nicole` |
| `--sample-rate` | `KOKORO_SAMPLE_RATE` | Output sample rate | `24000` |
| `--port` | `PORT` | Server port | `5000` |

Example using arguments:
```bash
python server.py --port 5001 --voice af_bella
```

## Voice catalog

Available voices are read from `VOICES.md`. Each entry exposes a `name` and `lang_code` to the frontend dropdown. Update that file to add or remove voices without touching code.

## How it works

- `/api/models` proxies LM Studio to list available model IDs.
- `/api/chat` sends the conversation to LM Studio, gets a text reply, and immediately synthesizes audio for the selected voice.
- `/api/voices` returns the Kokoro voices parsed from `VOICES.md`.
- `/api/tts` runs Kokoro TTS for arbitrary text (used by the “Speak” buttons).

Local chat history, selected model, voice, and temperature are cached in `localStorage` so your settings persist between refreshes.
