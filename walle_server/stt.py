"""
stt.py — Speech-to-Text module
Uses faster-whisper (tiny model) for low-latency transcription on M1 Mac.
Model is loaded once at startup and reused across requests.
"""

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_SIZE        = "tiny"       # tiny | base | small  (tiny ~300ms on M1)
COMPUTE_TYPE      = "int8"       # int8 = fastest on CPU, good accuracy
LANGUAGE          = "en"         # lock to English — faster, more accurate
MIN_CONFIDENCE    = 0.0          # whisper doesn't give per-word confidence
                                 # we use avg log-prob instead
MIN_LOG_PROB      = -1.2         # filter out very uncertain transcripts
BEAM_SIZE         = 1            # 1 = greedy, fastest
VAD_FILTER        = True         # built-in VAD — skips silent chunks


class STTEngine:
    """
    Wrapper around faster-whisper.
    Usage:
        engine = STTEngine()
        result = engine.transcribe(pcm_bytes)
        # result = {"transcript": str, "confidence": float, "duration_ms": int}
    """

    def __init__(self):
        self._model = None

    def load(self):
        """Load model — call once at server startup."""
        logger.info(f"Loading faster-whisper [{MODEL_SIZE}] ...")
        t0 = time.time()

        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                MODEL_SIZE,
                device="cpu",          # M1 MPS not yet stable in faster-whisper
                compute_type=COMPUTE_TYPE,
            )
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

        elapsed = time.time() - t0
        logger.info(f"STT model loaded in {elapsed:.2f}s")

    # ------------------------------------------------------------------

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribe raw 16-bit PCM bytes.

        Args:
            pcm_bytes:   Raw audio as 16-bit little-endian PCM
            sample_rate: Sample rate (default 16000 Hz from INMP441)

        Returns:
            {
              "transcript":  str,    # cleaned text, empty string if nothing heard
              "confidence":  float,  # 0.0-1.0 derived from avg log prob
              "duration_ms": int,    # inference time
            }
        """
        if self._model is None:
            raise RuntimeError("STT engine not loaded. Call load() first.")

        if not pcm_bytes or len(pcm_bytes) < 512:
            return _empty_result()

        # Convert raw bytes → float32 numpy array normalised to [-1, 1]
        audio_np = _pcm_bytes_to_float32(pcm_bytes, sample_rate)
        if audio_np is None:
            return _empty_result()

        t0 = time.time()

        try:
            segments, info = self._model.transcribe(
                audio_np,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                },
            )

            # Collect all segments
            texts = []
            log_probs = []

            for seg in segments:
                text = seg.text.strip()
                if text:
                    texts.append(text)
                    log_probs.append(seg.avg_logprob)

        except Exception as e:
            logger.error(f"Whisper inference error: {e}")
            return _empty_result()

        elapsed_ms = int((time.time() - t0) * 1000)

        if not texts:
            logger.debug(f"STT: no speech detected ({elapsed_ms}ms)")
            return _empty_result(elapsed_ms)

        transcript = " ".join(texts).strip()
        avg_log_prob = float(np.mean(log_probs)) if log_probs else -2.0

        # Convert log-prob to a 0–1 confidence score
        # log_prob of 0.0 = perfect, -1.0 = okay, < -1.2 = uncertain
        confidence = _log_prob_to_confidence(avg_log_prob)

        logger.info(
            f"STT [{elapsed_ms}ms] conf={confidence:.2f} | \"{transcript}\""
        )

        # Gate on minimum confidence
        if avg_log_prob < MIN_LOG_PROB:
            logger.debug(
                f"STT: transcript rejected (log_prob={avg_log_prob:.2f} < {MIN_LOG_PROB})"
            )
            return _empty_result(elapsed_ms)

        return {
            "transcript":  transcript,
            "confidence":  confidence,
            "duration_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm_bytes_to_float32(pcm_bytes: bytes, sample_rate: int):
    """Convert raw 16-bit PCM bytes to float32 numpy array."""
    try:
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Sanity check: mostly silence? (RMS < 0.001)
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        if rms < 0.0005:
            logger.debug(f"STT: audio too quiet (RMS={rms:.5f}), skipping")
            return None

        return audio_float

    except Exception as e:
        logger.error(f"PCM conversion error: {e}")
        return None


def _log_prob_to_confidence(log_prob: float) -> float:
    """
    Map whisper avg_logprob to [0, 1] confidence.
    Empirically: 0.0 = perfect, -0.5 = good, -1.0 = okay, -2.0 = bad
    """
    # Clamp and normalise: treat -2.0 as 0% and 0.0 as 100%
    clamped = max(-2.0, min(0.0, log_prob))
    return round(1.0 + (clamped / 2.0), 3)


def _empty_result(duration_ms: int = 0) -> dict:
    return {
        "transcript":  "",
        "confidence":  0.0,
        "duration_ms": duration_ms,
    }
    