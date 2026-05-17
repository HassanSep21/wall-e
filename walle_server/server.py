"""
server.py — WALL-E Python Backend Server
Flask server that receives audio from ESP32, runs STT + NLP, plays sounds,
and returns a JSON command response.

Run with:
    python server.py

Make sure you are connected to the ESP32_TEST WiFi AP first.
Your IP on that network should be 192.168.4.2 (printed on startup).
"""

import logging
import socket
import struct
import time
import json

from flask import Flask, request, jsonify

from stt          import STTEngine
from nlp          import NLPPipeline
from sound_engine import SoundEngine

# ---------------------------------------------------------------------------
# Logging — clean, timestamped output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("walle.server")

# Silence noisy third-party loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST            = "0.0.0.0"
PORT            = 5001
MIC_SAMPLE_RATE = 16000          # must match ESP32 I2S config
MAX_AUDIO_BYTES = 256_000        # ~8s at 16kHz 16-bit — hard cap
MIN_AUDIO_BYTES = 1_024          # reject tiny garbage payloads

# ---------------------------------------------------------------------------
# Module instances (singletons)
# ---------------------------------------------------------------------------
stt_engine    = STTEngine()
nlp_pipeline  = NLPPipeline()
sound_engine  = SoundEngine()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """
    ESP32 calls this on boot to verify the Python server is reachable.
    Returns server's local IP so ESP32 can confirm routing.
    """
    local_ip = _get_local_ip()
    return jsonify({
        "status": "ok",
        "server_ip": local_ip,
        "timestamp": time.time(),
    }), 200


@app.route("/process", methods=["POST"])
def process_audio():
    """
    Main endpoint. Receives raw 16-bit PCM audio from ESP32.
    Pipeline: PCM bytes → STT → NLP → Sound → JSON response

    Request:
        Content-Type: application/octet-stream
        Body: raw 16-bit little-endian PCM at 16kHz

    Response JSON:
        {
          "emotion":            str,   // happy|sad|angry|curious|surprised|neutral
          "command":            str,   // dance|wave|stop|spin|look|sleep|idle
          "transcript":         str,   // what was heard
          "confidence":         float, // STT confidence 0.0–1.0
          "emotion_confidence": float,
          "command_matched":    bool,
          "processing_ms":      int,   // total server-side latency
          "error":              null   // or error string
        }
    """
    t_start = time.time()

    # ── 1. Validate incoming audio ───────────────────────────────────────
    pcm_bytes = request.data

    if not pcm_bytes:
        logger.warning("/process called with empty body")
        return _error_response("No audio data received", 400)

    byte_count = len(pcm_bytes)
    logger.info(f"/process received {byte_count} bytes "
                f"({byte_count / (MIC_SAMPLE_RATE * 2):.2f}s of audio)")

    if byte_count < MIN_AUDIO_BYTES:
        return _error_response(f"Audio too short ({byte_count} bytes)", 400)

    if byte_count > MAX_AUDIO_BYTES:
        logger.warning(f"Audio truncated from {byte_count} to {MAX_AUDIO_BYTES} bytes")
        pcm_bytes = pcm_bytes[:MAX_AUDIO_BYTES]

    # ── 2. Speech-to-Text ────────────────────────────────────────────────
    stt_result = stt_engine.transcribe(pcm_bytes, sample_rate=MIC_SAMPLE_RATE)
    transcript  = stt_result["transcript"]
    stt_conf    = stt_result["confidence"]

    logger.info(f"STT → \"{transcript}\" (conf={stt_conf:.2f}, "
                f"{stt_result['duration_ms']}ms)")

    # ── 3. NLP pipeline ─────────────────────────────────────────────────
    nlp_result  = nlp_pipeline.process(transcript, stt_conf, pcm_bytes)
    emotion     = nlp_result["emotion"]
    command     = nlp_result["command"]
    emo_conf    = nlp_result["emotion_confidence"]
    cmd_matched = nlp_result["command_matched"]

    logger.info(f"NLP  → emotion={emotion}({emo_conf:.2f}) command={command}")

    # ── 4. Sound playback (non-blocking) ────────────────────────────────
    sound_engine.play(command=command, emotion=emotion)

    # ── 5. Build + return response ───────────────────────────────────────
    processing_ms = int((time.time() - t_start) * 1000)

    response_payload = {
        "emotion":            emotion,
        "command":            command,
        "transcript":         transcript,
        "confidence":         round(stt_conf, 3),
        "emotion_confidence": round(emo_conf, 3),
        "command_matched":    cmd_matched,
        "processing_ms":      processing_ms,
        "error":              None,
    }

    logger.info(
        f"Response → {json.dumps(response_payload, ensure_ascii=False)} "
        f"[{processing_ms}ms total]"
    )

    return jsonify(response_payload), 200


@app.route("/status", methods=["GET"])
def status():
    """
    Health + capability status endpoint.
    Useful for debugging from a browser.
    """
    return jsonify({
        "status":          "running",
        "stt_ready":       stt_engine._model is not None,
        "nlp_ready":       nlp_pipeline._classifier is not None,
        "sound_ready":     sound_engine._initialized,
        "server_ip":       _get_local_ip(),
        "uptime_s":        round(time.time() - _start_time, 1),
    }), 200


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error_response(message: str, status_code: int):
    """Return a well-formed error JSON so ESP32 can always parse the response."""
    return jsonify({
        "emotion":            "neutral",
        "command":            "idle",
        "transcript":         "",
        "confidence":         0.0,
        "emotion_confidence": 0.0,
        "command_matched":    False,
        "processing_ms":      0,
        "error":              message,
    }), status_code


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    """Get this machine's IP on the currently connected network."""
    try:
        # This trick works without actually sending a packet
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.4.1", 80))   # connect toward ESP32 AP gateway
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
_start_time = time.time()


def startup():
    """Load all models before accepting requests."""
    print()
    print("=" * 60)
    print("  WALL-E Python Backend Server")
    print("=" * 60)

    print("\n[1/3] Loading STT model (faster-whisper tiny)...")
    stt_engine.load()
    print("      ✓ STT ready")

    print("\n[2/3] Loading NLP model (emotion-distilroberta)...")
    nlp_pipeline.load()
    print("      ✓ NLP ready")

    print("\n[3/3] Initializing sound engine (pygame)...")
    sound_engine.load()
    print("      ✓ Sound ready")

    local_ip = _get_local_ip()

    print()
    print("=" * 60)
    print(f"  Server IP on ESP32 network : {local_ip}")
    print(f"  Listening on               : http://0.0.0.0:{PORT}")
    print()
    print("  IMPORTANT: If IP is not 192.168.4.2, update")
    print("  PYTHON_SERVER_IP in the ESP32 sketch.")
    print("=" * 60)
    print()

    if local_ip != "192.168.4.2":
        print(f"  ⚠️  WARNING: Your IP is {local_ip}, not 192.168.4.2")
        print(f"     Update #define PYTHON_SERVER_IP \"{local_ip}\"")
        print(f"     in walle_esp32.ino and re-upload.")
        print()


if __name__ == "__main__":
    startup()
    # threaded=False: we process one audio request at a time
    # (ESP32 sends sequentially anyway)
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,    # must be False — reloader breaks model loading
    )
    