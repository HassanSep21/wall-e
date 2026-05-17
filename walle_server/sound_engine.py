"""
sound_engine.py — WALL-E Sound Engine
Plays WALL-E style sounds on MacOS via pygame.mixer.
Sound selection is based on (command, emotion) tuple.
Playback is non-blocking — sound plays while HTTP response is already sent.
"""

import os
import logging
import threading
import time
import pygame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# ---------------------------------------------------------------------------
# Sound map: (command, emotion) → filename
# Lookup is attempted in order:
#   1. Exact (command, emotion) match
#   2. ("idle", emotion) fallback
#   3. ("idle", "neutral") final fallback
# ---------------------------------------------------------------------------
SOUND_MAP: dict[tuple[str, str], str] = {
    # Command-specific sounds
    ("dance",   "happy"):     "walle_excited.wav",
    ("dance",   "neutral"):   "walle_excited.wav",
    ("dance",   "surprised"): "walle_excited.wav",
    ("wave",    "happy"):     "walle_greeting.wav",
    ("wave",    "neutral"):   "walle_greeting.wav",
    ("wave",    "surprised"): "walle_greeting.wav",
    ("stop",    "sad"):       "walle_sad.wav",
    ("stop",    "angry"):     "walle_concern.wav",
    ("stop",    "neutral"):   "walle_neutral.wav",
    ("spin",    "curious"):   "walle_curious.wav",
    ("spin",    "happy"):     "walle_excited.wav",
    ("spin",    "neutral"):   "walle_curious.wav",
    ("look",    "curious"):   "walle_curious.wav",
    ("look",    "neutral"):   "walle_neutral.wav",
    ("sleep",   "sad"):       "walle_tired.wav",
    ("sleep",   "neutral"):   "walle_tired.wav",

    # Idle + emotion fallbacks (used when no command matched)
    ("idle",    "happy"):     "walle_happy_chirp.wav",
    ("idle",    "sad"):       "walle_sad_beep.wav",
    ("idle",    "angry"):     "walle_concern.wav",
    ("idle",    "curious"):   "walle_curious.wav",
    ("idle",    "surprised"): "walle_excited.wav",
    ("idle",    "neutral"):   "walle_neutral.wav",
}

FALLBACK_SOUND = "walle_neutral.wav"


class SoundEngine:
    """
    Manages pygame mixer and non-blocking sound playback.

    Usage:
        engine = SoundEngine()
        engine.load()
        engine.play(command="dance", emotion="happy")  # non-blocking
    """

    def __init__(self):
        self._initialized  = False
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}
        self._lock         = threading.Lock()

    # ------------------------------------------------------------------

    def load(self):
        """Initialize pygame mixer. Call once at startup."""
        try:
            pygame.mixer.pre_init(
                frequency=22050,
                size=-16,       # signed 16-bit
                channels=1,
                buffer=512,
            )
            pygame.mixer.init()
            self._initialized = True
            logger.info(
                f"Sound engine initialized "
                f"(freq={pygame.mixer.get_init()[0]}Hz, "
                f"channels={pygame.mixer.get_init()[2]})"
            )
        except Exception as e:
            logger.error(f"pygame mixer init failed: {e}")
            self._initialized = False

        # Pre-load all sounds into cache
        self._preload_sounds()

    # ------------------------------------------------------------------

    def play(self, command: str, emotion: str):
        """
        Select and play the appropriate sound asynchronously.
        Returns immediately — playback happens in a background thread.

        Args:
            command: e.g. "dance", "wave", "idle"
            emotion: e.g. "happy", "sad", "neutral"
        """
        if not self._initialized:
            logger.warning("Sound engine not initialized, skipping playback.")
            return

        filename = _select_sound(command, emotion)
        thread   = threading.Thread(
            target=self._play_file,
            args=(filename,),
            daemon=True,
            name=f"sound-{filename}",
        )
        thread.start()

    # ------------------------------------------------------------------

    def _play_file(self, filename: str):
        """Internal: play a WAV file (runs in background thread)."""
        with self._lock:
            sound = self._sound_cache.get(filename)

            if sound is None:
                path = os.path.join(SOUNDS_DIR, filename)
                if not os.path.isfile(path):
                    logger.error(f"Sound file not found: {path}")
                    return
                try:
                    sound = pygame.mixer.Sound(path)
                    self._sound_cache[filename] = sound
                except Exception as e:
                    logger.error(f"Failed to load sound {filename}: {e}")
                    return

        try:
            # Stop any currently playing sound (one sound at a time for V1)
            pygame.mixer.stop()
            channel = sound.play()
            if channel:
                # Wait for playback to finish (inside daemon thread — safe)
                while channel.get_busy():
                    time.sleep(0.02)
        except Exception as e:
            logger.error(f"Playback error ({filename}): {e}")

    # ------------------------------------------------------------------

    def _preload_sounds(self):
        """Load all WAV files into memory at startup to avoid first-play delay."""
        if not self._initialized:
            return

        loaded = 0
        missing = 0

        for filename in set(SOUND_MAP.values()):
            path = os.path.join(SOUNDS_DIR, filename)
            if os.path.isfile(path):
                try:
                    self._sound_cache[filename] = pygame.mixer.Sound(path)
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Could not preload {filename}: {e}")
                    missing += 1
            else:
                logger.warning(f"Sound file missing: {path}  (run generate_sounds.py)")
                missing += 1

        logger.info(f"Sound cache: {loaded} loaded, {missing} missing")

    # ------------------------------------------------------------------

    def stop(self):
        """Stop all playback and quit mixer cleanly."""
        if self._initialized:
            pygame.mixer.stop()
            pygame.mixer.quit()
            self._initialized = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_sound(command: str, emotion: str) -> str:
    """
    Resolve (command, emotion) to a filename using SOUND_MAP.
    Fallback chain:
      1. exact match
      2. ("idle", emotion)
      3. FALLBACK_SOUND
    """
    key = (command, emotion)
    if key in SOUND_MAP:
        return SOUND_MAP[key]

    idle_key = ("idle", emotion)
    if idle_key in SOUND_MAP:
        logger.debug(f"Sound: no exact match for {key}, using {idle_key}")
        return SOUND_MAP[idle_key]

    logger.debug(f"Sound: no match for {key} or {idle_key}, using fallback")
    return FALLBACK_SOUND
