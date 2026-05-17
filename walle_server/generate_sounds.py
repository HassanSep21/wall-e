"""
generate_sounds.py
Generates robotic WALL-E-style placeholder sounds using numpy.
Run once before starting the server: python generate_sounds.py
"""

import numpy as np
import soundfile as sf
import os

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)

SR = 22050  # sample rate


def sine(freq, duration, sr=SR, amp=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def square(freq, duration, sr=SR, amp=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sign(np.sin(2 * np.pi * freq * t)) * amp).astype(np.float32)


def silence(duration, sr=SR):
    return np.zeros(int(sr * duration), dtype=np.float32)


def envelope(audio, attack=0.01, release=0.05, sr=SR):
    """Apply attack/release envelope to avoid clicks."""
    a = int(sr * attack)
    r = int(sr * release)
    audio = audio.copy()
    if a > 0:
        audio[:a] *= np.linspace(0, 1, a)
    if r > 0 and r <= len(audio):
        audio[-r:] *= np.linspace(1, 0, r)
    return audio


def fm_beep(carrier, modulator, mod_depth, duration, amp=0.4, sr=SR):
    """FM synthesis for richer robotic tones."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mod = mod_depth * np.sin(2 * np.pi * modulator * t)
    wave = amp * np.sin(2 * np.pi * carrier * t + mod)
    return wave.astype(np.float32)


def chirp(f_start, f_end, duration, amp=0.4, sr=SR):
    """Frequency sweep."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = np.linspace(f_start, f_end, len(t))
    phase = np.cumsum(2 * np.pi * freqs / sr)
    return (amp * np.sin(phase)).astype(np.float32)


def save(name, audio, sr=SR):
    path = os.path.join(SOUNDS_DIR, name)
    # Clamp to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr)
    print(f"  ✓ {name}")


def generate_all():
    print("Generating WALL-E style robotic sounds...")

    # --- walle_excited.wav ---
    # Rising chirps + fast beeps = excitement
    parts = []
    for f in [400, 600, 800, 1000, 1200]:
        parts.append(envelope(sine(f, 0.08, amp=0.45)))
        parts.append(silence(0.03))
    parts.append(chirp(300, 1400, 0.25, amp=0.4))
    parts.append(silence(0.05))
    parts.append(envelope(fm_beep(900, 15, 3, 0.3, amp=0.4)))
    save("walle_excited.wav", np.concatenate(parts))

    # --- walle_greeting.wav ---
    # Friendly two-tone rising beep
    parts = [
        envelope(sine(440, 0.15, amp=0.4)),
        silence(0.04),
        envelope(sine(660, 0.20, amp=0.4)),
        silence(0.06),
        envelope(chirp(500, 800, 0.18, amp=0.35)),
    ]
    save("walle_greeting.wav", np.concatenate(parts))

    # --- walle_sad.wav ---
    # Slow descending tone, wobbly
    t = np.linspace(0, 0.6, int(SR * 0.6), endpoint=False)
    wobble = 0.3 * np.sin(2 * np.pi * 4 * t)  # 4 Hz wobble
    freq_env = np.linspace(600, 280, len(t))
    phase = np.cumsum(2 * np.pi * (freq_env + wobble * 30) / SR)
    sad = (0.35 * np.sin(phase)).astype(np.float32)
    sad = envelope(sad, attack=0.05, release=0.15)
    parts = [sad, silence(0.08), envelope(sine(250, 0.3, amp=0.25))]
    save("walle_sad.wav", np.concatenate(parts))

    # --- walle_curious.wav ---
    # Rising question-like chirp, then small beep
    parts = [
        envelope(chirp(350, 750, 0.22, amp=0.38)),
        silence(0.05),
        envelope(chirp(750, 900, 0.10, amp=0.3)),
        silence(0.06),
        envelope(sine(800, 0.08, amp=0.3)),
    ]
    save("walle_curious.wav", np.concatenate(parts))

    # --- walle_tired.wav ---
    # Slow descending, low energy
    parts = [
        envelope(chirp(500, 200, 0.5, amp=0.3)),
        silence(0.1),
        envelope(sine(180, 0.3, amp=0.2)),
    ]
    save("walle_tired.wav", np.concatenate(parts))

    # --- walle_happy_chirp.wav ---
    # Short upbeat sequence
    freqs = [523, 659, 784]  # C, E, G major chord ascending
    parts = []
    for f in freqs:
        parts.append(envelope(sine(f, 0.10, amp=0.4)))
        parts.append(silence(0.02))
    parts.append(envelope(fm_beep(784, 10, 2, 0.15, amp=0.35)))
    save("walle_happy_chirp.wav", np.concatenate(parts))

    # --- walle_sad_beep.wav ---
    # Minor descending
    freqs = [392, 349, 294]  # G, F, D descending
    parts = []
    for f in freqs:
        parts.append(envelope(sine(f, 0.14, amp=0.35)))
        parts.append(silence(0.03))
    save("walle_sad_beep.wav", np.concatenate(parts))

    # --- walle_concern.wav ---
    # Rapid low buzzy pulses = worried/alert
    parts = []
    for _ in range(4):
        parts.append(envelope(square(180, 0.07, amp=0.28)))
        parts.append(silence(0.04))
    parts.append(envelope(chirp(200, 150, 0.2, amp=0.3)))
    save("walle_concern.wav", np.concatenate(parts))

    # --- walle_neutral.wav ---
    # Single clean mid beep
    parts = [
        envelope(sine(520, 0.12, amp=0.38)),
        silence(0.04),
        envelope(sine(520, 0.08, amp=0.28)),
    ]
    save("walle_neutral.wav", np.concatenate(parts))

    print(f"\nAll sounds saved to: {SOUNDS_DIR}")
    print("Tip: Replace any .wav with a real WALL-E clip of the same filename.")
    print("     Search: 'WALL-E sound effects pack' on freesound.org or mixkit.co")


if __name__ == "__main__":
    generate_all()
    