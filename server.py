import base64
import io
import os
import re
import threading
import time
import wave
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from kokoro import KModel, KPipeline
from openai import OpenAI

# ---------- Configuration ----------
def parse_args():
    parser = argparse.ArgumentParser(description="LM-Studio Chatbot with Kokoro TTS")
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"), help="Base URL for the LM Studio API (default: http://127.0.0.1:1234/v1)")
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", "lm-studio"), help="API key for LM Studio (default: lm-studio)")
    parser.add_argument("--model-name", default=os.getenv("LMSTUDIO_MODEL_NAME"), help="Force a specific model ID; omit to auto-pick the first loaded model")
    parser.add_argument("--repo-id", default=os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M"), help="Hugging Face repo for Kokoro weights (default: hexgrad/Kokoro-82M)")
    parser.add_argument("--lang", default=os.getenv("KOKORO_LANG", "a"), help="Default language code used when parsing VOICES.md (single letter, default: a)")
    parser.add_argument("--voice", default=os.getenv("KOKORO_VOICE"), help="Default voice name; must exist in VOICES.md")
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("KOKORO_SAMPLE_RATE", "24000")), help="Output sample rate (default: 24000)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")), help="Server port (default: 5000)")
    return parser.parse_args()

args = parse_args()

LMSTUDIO_BASE_URL = args.base_url
LMSTUDIO_API_KEY = args.api_key
LMSTUDIO_MODEL_NAME = args.model_name
KOKORO_REPO_ID = args.repo_id
KOKORO_DEFAULT_LANG = args.lang
SAMPLE_RATE = args.sample_rate
PORT = args.port
VOICES_PATH = Path(__file__).with_name("VOICES.md")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")


def _parse_voice_catalog(path: Path, fallback_lang: str) -> Dict[str, Dict[str, str]]:
    voices: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return voices

    def _detect_gender(name: str, cells: list[str]) -> str:
        combined = " ".join(cells)
        if "🚺" in combined or "♀" in combined:
            return "female"
        if "🚹" in combined or "♂" in combined:
            return "male"
        prefix = name.split("_", 1)[0].lower()
        if prefix.endswith("f"):
            return "female"
        if prefix.endswith("m"):
            return "male"
        return "unknown"

    lang_code = fallback_lang
    lang_re = re.compile(r"lang_code='([a-z])'", re.IGNORECASE)

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            lang_match = lang_re.search(line)
            if lang_match:
                lang_code = lang_match.group(1)
                continue

            if not line.startswith("|") or line.startswith("| Name") or line.startswith("| ----"):
                continue

            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not cells:
                continue

            name_cell = cells[0]
            # Remove markdown emphasis/escape characters (e.g., "**af\_heart**" -> "af_heart")
            name = re.sub(r"[`*]", "", name_cell).replace("\\_", "_").strip()
            if not name or name.lower() == "name":
                continue

            gender = _detect_gender(name, cells)

            voices[name] = {"lang": lang_code, "gender": gender}

    return voices


# ---------- App + Clients ----------
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"
kokoro_model = KModel(repo_id=KOKORO_REPO_ID).to(device).eval()
tts_lock = threading.Lock()
pipelines: Dict[str, KPipeline] = {}
_model_cache: dict[str, float | List[str]] = {"ts": 0.0, "models": []}
MODEL_CACHE_TTL = 30.0  # seconds

voice_catalog = _parse_voice_catalog(VOICES_PATH, KOKORO_DEFAULT_LANG)
PREFERRED_DEFAULT_VOICE = "af_nicole"
cli_default_voice = args.voice
if cli_default_voice and cli_default_voice in voice_catalog:
    DEFAULT_VOICE = cli_default_voice
elif PREFERRED_DEFAULT_VOICE in voice_catalog:
    DEFAULT_VOICE = PREFERRED_DEFAULT_VOICE
else:
    DEFAULT_VOICE = next(iter(voice_catalog.keys()), env_default_voice or PREFERRED_DEFAULT_VOICE)
    if DEFAULT_VOICE not in voice_catalog:
        voice_catalog[DEFAULT_VOICE] = {"lang": KOKORO_DEFAULT_LANG, "gender": "unknown"}


def get_voice_info(requested: Optional[str]) -> tuple[str, str]:
    voice = (requested or DEFAULT_VOICE or "").strip()
    info = voice_catalog.get(voice)
    if not info:
        raise ValueError(f"Voice '{voice}' is not available.")
    return voice, info["lang"]


def get_pipeline(lang_code: str) -> KPipeline:
    pipeline = pipelines.get(lang_code)
    if pipeline is None:
        pipeline = KPipeline(
            lang_code=lang_code, repo_id=KOKORO_REPO_ID, model=kokoro_model
        )
        pipelines[lang_code] = pipeline
    return pipeline


def _audio_to_wav_base64(audio: np.ndarray) -> str:
    """Convert Kokoro float audio to base64 wav bytes."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).flatten()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return base64.b64encode(buf.getvalue()).decode("ascii")


def strip_markdown(text: str) -> str:
    """
    Remove common markdown styling so it isn't read aloud by TTS.
    """
    # 1. Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 2. Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 3. Remove bold/italic: **outer **inner**** or __outer __inner____
    # We do it a couple of times to handle some nesting
    for _ in range(2):
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # 4. Remove links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # 5. Remove headers: # Header
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # 6. Remove list markers: * item or 1. item
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # 7. Remove blockquotes: > quote
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)
    # 8. Final cleanup: strip extra whitespace
    return text.strip()


def synthesize_audio(
    text: str, voice_name: Optional[str] = None
) -> Tuple[str, str]:
    resolved_voice, lang_code = get_voice_info(voice_name)

    # Remove markdown formatting before TTS
    text = strip_markdown(text or "")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("No text provided to synthesize.")

    last_error: Optional[Exception] = None
    for attempt in (0, 1):
        pipeline = get_pipeline(lang_code)

        try:
            with tts_lock:
                segments: List[np.ndarray] = []
                for chunk in pipeline(clean_text, voice=resolved_voice):
                    audio = chunk.audio
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    arr = np.asarray(audio, dtype=np.float32).flatten()
                    if arr.size:
                        segments.append(arr)
                if not segments:
                    raise RuntimeError("Kokoro returned empty audio.")
                audio = np.concatenate(segments)
            return _audio_to_wav_base64(audio), resolved_voice
        except Exception as exc:  # pragma: no cover - safety net for runtime TTS failures
            last_error = exc
            # Reset the cached pipeline in case it got into a bad state for this language.
            pipelines.pop(lang_code, None)
            if attempt == 0:
                continue
            error_msg = f"TTS failed for voice '{resolved_voice}': {exc}"
            app.logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

    raise RuntimeError(
        f"TTS failed for voice '{resolved_voice}': {last_error}"
    ) from last_error


def _fetch_lmstudio_models() -> List[str]:
    """Query LM Studio for available model IDs."""
    try:
        response = client.models.list()
    except Exception as exc:
        app.logger.warning("Failed to fetch LM Studio models: %s", exc)
        return []

    data = getattr(response, "data", []) or []
    return [getattr(model, "id", "") for model in data if getattr(model, "id", "")]


def available_models(force_refresh: bool = False) -> List[str]:
    """Return cached list of LM Studio models or query again."""
    if LMSTUDIO_MODEL_NAME:
        return [LMSTUDIO_MODEL_NAME]

    now = time.monotonic()
    cached = _model_cache["models"]
    ts = _model_cache["ts"]
    if not force_refresh and cached and now - ts < MODEL_CACHE_TTL:
        return cached  # type: ignore[return-value]

    models = _fetch_lmstudio_models()
    if models:
        _model_cache["models"] = models
        _model_cache["ts"] = now
        return models

    return cached or []  # type: ignore[return-value]


def resolve_model_name(requested: Optional[str] = None) -> str:
    """Pick a model for the current completion request."""
    if requested:
        return requested

    models = available_models()
    if not models:
        raise RuntimeError("No LM Studio model is currently loaded.")
    return models[0]


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True) or {}
    temperature = float(payload.get("temperature", 0.7))
    tts_enabled = payload.get("tts_enabled", True)
    requested_voice = payload.get("voice")

    if tts_enabled:
        try:
            resolved_voice, _ = get_voice_info(requested_voice)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        resolved_voice = requested_voice or DEFAULT_VOICE

    try:
        model_name = resolve_model_name(payload.get("model"))
        previous_response_id = payload.get("previous_response_id")
        system_prompt = payload.get("system_prompt")
        reminder_prompt = payload.get("reminder_prompt")
        input_text = payload.get("input_text", "")

        if not input_text and not previous_response_id:
            return jsonify({"error": "input_text is required for new chats"}), 400

        # Build the combined instructions (system prompt + optional reminder)
        instructions = system_prompt or ""
        if reminder_prompt:
            instructions = f"{instructions}\n\n[REMINDER]: {reminder_prompt}".strip()

        # Prepare payload for stateful API
        responses_payload = {
            "model": model_name,
            "input": input_text,
            "temperature": temperature,
            "stream": False
        }
        if previous_response_id:
            responses_payload["previous_response_id"] = previous_response_id
        if instructions:
            responses_payload["instructions"] = instructions

        # Use LM Studio's /v1/responses endpoint
        responses_url = f"{LMSTUDIO_BASE_URL}/responses"
        resp = requests.post(responses_url, json=responses_payload, timeout=60)
        resp.raise_for_status()

        completion_data = resp.json()

        # Extract content from LM Studio /v1/responses format
        # Based on observed schema: output[0] -> content[0] -> text
        content = ""
        output_list = completion_data.get("output", [])
        if output_list and isinstance(output_list, list) and len(output_list) > 0:
            first_output = output_list[0]

            # Case 1: content[0]['text'] (Message role schema)
            content_list = first_output.get("content", [])
            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                first_content = content_list[0]
                content = first_content.get("text", "")

            # Case 2: text['content'] (Alternative schema)
            if not content:
                text_obj = first_output.get("text", {})
                if isinstance(text_obj, dict):
                    content = text_obj.get("content", "")
                elif isinstance(text_obj, str):
                    content = text_obj

            # Case 3: output_text field (Legacy/experimental)
            if not content:
                content = first_output.get("output_text", "")

        # Final fallback - check top-level if nothing found in output list
        if not content:
            content = completion_data.get("output_text", "")

        response_id = completion_data.get("id", "")

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.error("LLM failure: %s", exc)
        return jsonify({"error": str(exc)}), 500

    # Only synthesize audio if TTS is enabled
    audio_b64 = None
    if tts_enabled:
        try:
            audio_b64, resolved_voice = synthesize_audio(content, resolved_voice)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    result = {
        "content": content,
        "model": model_name,
        "voice": resolved_voice,
        "response_id": response_id
    }
    if audio_b64:
        result["audio"] = audio_b64

    return jsonify(result)


@app.route("/api/tts", methods=["POST"])
def tts():
    payload = request.get_json(force=True) or {}
    text = payload.get("text", "")
    voice = payload.get("voice")

    try:
        audio_b64, resolved_voice = synthesize_audio(text, voice)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"audio": audio_b64, "voice": resolved_voice})


@app.route("/api/models", methods=["GET"])
def list_models():
    models = available_models(force_refresh=True)
    return jsonify({"models": models})


@app.route("/api/voices", methods=["GET"])
def list_voices():
    voices = [
        {"name": name, "lang_code": meta["lang"], "gender": meta.get("gender", "unknown")}
        for name, meta in sorted(voice_catalog.items())
    ]
    return jsonify({"voices": voices, "default": DEFAULT_VOICE})


@app.route("/")
def root():
    return app.send_static_file("index.html")


@app.errorhandler(404)
def not_found(_):
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
