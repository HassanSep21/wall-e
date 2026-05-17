# WALL-E AI Robot

A voice-interactive robot built on ESP32-S3, inspired by the WALL-E character. The robot listens for spoken commands, classifies the speaker's emotion using a fine-tuned transformer model, and responds with facial expressions, motion sequences, and character sounds — all processed locally with no internet dependency during operation.

---

## Repository Structure

```
walle/
├── walle_esp32/
│   └── walle_esp32.ino          # Arduino sketch (ESP32-S3, all hardware modules)
│
└── walle_server/
    ├── server.py                # Flask HTTP server — main entry point
    ├── stt.py                   # Speech-to-Text (faster-whisper)
    ├── nlp.py                   # Emotion classification + command routing
    ├── sound_engine.py          # Sound playback (pygame)
    ├── generate_sounds.py       # Generates robotic placeholder WAV files
    ├── train_emotion_model.py   # Fine-tunes DistilBERT on emotion dataset
    ├── train_fusion.py          # Trains multimodal fusion MLP
    ├── requirements.txt
    ├── sounds/                  # WAV files (generated or replaced with real clips)
    └── models/                  # Trained model weights (git-ignored, regenerable)
        ├── text_emotion/        # Fine-tuned DistilBERT weights
        ├── fusion_mlp.pt        # Multimodal fusion MLP weights
        └── fusion_config.json   # Feature dimensions and normalization stats
```

---

## Hardware

### Components

| Component | Model | Role |
|---|---|---|
| Microcontroller | ESP32-S3 Dev Module | Main controller, WiFi AP, I2S |
| Microphone | INMP441 | I2S digital mic, audio capture |
| Display | ST7735 TFT (128x160) | Facial expression renderer |
| Motor driver | L298N | Controls two DC drive motors |
| Servo x3 | SG90 / MG90S | Head pan, left arm, right arm |
| Speaker | MAX98357 | I2S amplifier (currently unused — see note) |
| Laptop | MacBook M1 | Runs Python inference server |

> The MAX98357 speaker module on this unit is non-functional. Audio output is routed to the laptop's speakers via pygame instead.

### Pin Configuration

```
INMP441 Microphone
  SCK  → GPIO 12
  WS   → GPIO 13
  DIN  → GPIO 11

ST7735 TFT Display
  CS   → GPIO 17
  DC   → GPIO 18
  RST  → GPIO  8
  SCLK → GPIO 46
  MOSI → GPIO  3
  BL   → GPIO 16

MAX98357 Amplifier
  BCLK → GPIO 21
  LRC  → GPIO 47
  DIN  → GPIO 20

Servos
  Servo 1 (Head)      → GPIO 38
  Servo 2 (Left arm)  → GPIO 39
  Servo 3 (Right arm) → GPIO 40

L298N Motor Driver
  IN1 → GPIO 7
  IN2 → GPIO 6
  IN3 → GPIO 5
  IN4 → GPIO 4
```

---

## System Architecture

### Network Topology

The ESP32 runs as a WiFi Access Point (`ESP32_TEST`). The laptop connects to this network. The Python server runs on the laptop at `192.168.4.2:5001`. The ESP32 POSTs audio to the server and receives JSON responses. No external internet connection is required during operation.

```
┌──────────────────────┐    WiFi AP (192.168.4.x)    ┌───────────────────────┐
│     ESP32-S3         │◄──────────────────────────► │   Laptop (Python)     │
│     192.168.4.1      │    HTTP POST /process       │   192.168.4.2:5001    │
│                      │    ← JSON response          │                       │
│  - Audio capture     │                             │  - faster-whisper STT │
│  - VAD detection     │                             │  - DistilBERT emotion │
│  - Servo / motor     │                             │  - Command routing    │
│  - TFT expressions   │                             │  - Sound playback     │
│  - Web dashboard     │                             │                       │
└──────────────────────┘                             └───────────────────────┘
```

### Module Map

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            SYSTEM MODULES                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ESP32-S3                                                                  │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────────────────┐    │
│  │  M1: Audio    │───►│  M2: Comms    │◄──►│  M3: Command Parser      │    │
│  │  Capture      │    │  WiFi / HTTP  │    │  JSON → RobotIntent enum │    │
│  │  VAD + buffer │    │  HTTP client  │    └─────────────┬────────────┘    │
│  └───────────────┘    └───────────────┘                  │                 │
│                                                          │                 │
│                       ┌──────────────────────────────────▼──────────────┐  │
│                       │             M4: Action Dispatcher               │  │
│                       │         non-blocking state machine              │  │
│                       │   IDLE → LISTENING → SENDING → EXECUTING → IDLE │  │
│                       └────────┬──────────────┬──────────────┬──────────┘  │
│                                │              │              │             │
│                    ┌───────────▼──┐  ┌────────▼──────┐  ┌────▼──────────┐  │
│                    │  M5: Motion  │  │ M6: Expression│  │  M7: Web UI   │  │
│                    │  Controller  │  │ TFT Eye Engine│  │  Dashboard    │  │
│                    │ Servo+Motor  │  │ 6 emotions    │  │  AP server    │  │
│                    └──────────────┘  └───────────────┘  └───────────────┘  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Python Server (Laptop)                                                    │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐   │
│  │  M8: Audio    │───►│  M9: STT      │───►│  M10: NLP Pipeline        │   │
│  │  Receiver     │    │  faster-      │    │  M10a: DistilBERT         │   │
│  │  Flask /proc  │    │  whisper tiny │    │        emotion classifier │   │
│  └───────────────┘    └───────────────┘    │  M10b: Command rule eng.  │   │
│                                            └─────────────┬─────────────┘   │
│                                                          │                 │
│                            ┌─────────────────────────────▼──────────────┐  │
│                            │  M11: Sound Engine                         │  │
│                            │  pygame — (command, emotion) → WAV clip    │  │
│                            └────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### Full Data Flow

```
User speaks near WALL-E
        │
        ▼
INMP441 mic → I2S → ESP32 Core 0 (audioTask)
        │
        │  Web UI button pressed  (manual trigger)
        ▼
ESP32 buffers 2.5s of 16kHz 16-bit PCM  (~80 KB)
        │
        ▼
HTTP POST → 192.168.4.2:5001/process  (raw audio bytes)
        │
        ├── faster-whisper tiny → transcript text       (~200-400ms)
        │
        ├── Wake phrase check ("hey buddy")
        │       if not found → return idle, ESP32 does nothing
        │
        ├── DistilBERT emotion classifier → emotion label  (~150ms)
        │
        ├── Regex command rule engine → command string      (~1ms)
        │
        └── pygame → plays WAV clip on laptop speakers  (async, non-blocking)
                │
                ▼
        JSON response → ESP32
        {emotion, command, transcript, confidence}
                │
                ▼
        M3 parses → RobotIntent struct
                │
                ▼
        M4 state machine dispatches:
                ├── M6 → TFT redraws eyes immediately
                └── M5 → servo + motor sequence runs
                │
                ▼
        Sequence completes → return to IDLE → resume VAD listening

End-to-end latency: ~800ms – 1.5s
```

---

## The 5 Key AI Features

### 1. Offline Speech Recognition
The INMP441 microphone captures audio at 16kHz via I2S on a dedicated FreeRTOS task (Core 0). Audio capture is triggered either by the "Listen Now" button on the web dashboard or automatically via the wake phrase detection filter on the Python side. The 2.5-second audio buffer is transmitted over WiFi to the Python server where `faster-whisper` (CTranslate2-optimized Whisper) transcribes it using the `tiny` model in 200–400ms. The entire pipeline runs offline — no cloud API, no internet dependency.

### 2. Fine-tuned Emotion Classification
A DistilBERT model was fine-tuned from scratch on the `dair-ai/emotion` dataset (20,000 labeled English sentences, 6 emotion classes). Training ran for 3 epochs on Apple Silicon MPS. The resulting model achieves ~88% test accuracy and runs inference in ~150ms. This is not a pretrained model being called as a black box — the weights are trained specifically for this project and stored locally.

### 3. Multimodal Sentiment Architecture
The NLP pipeline is architected for multimodal emotion fusion. Text embeddings (CLS token, 768-dim) from DistilBERT are concatenated with acoustic features extracted by librosa (13 MFCC means, 13 MFCC standard deviations, pitch, RMS energy, zero-crossing rate — 29 dimensions total) and passed through a two-layer MLP fusion network. The current deployment uses text-only classification due to the domain gap between simulated and real acoustic training data; the fusion model is trained and present in the codebase as an architectural component.

### 4. Wake Phrase Detection + Context-aware Command Routing
A wake phrase filter ("hey buddy" and phonetic variants) gates the full inference pipeline on the Python side — audio without the trigger phrase is discarded after STT without invoking the NLP or sound engine. After wake phrase detection, a regex-based command rule engine maps the stripped transcript to 10 action categories (dance, wave, spin, look, sleep, stop, forward, backward, left, right). Rule ordering is deliberately prioritised — directional movement commands are matched before more general patterns to prevent ambiguity. Emotion and command are treated as independent outputs: emotion drives the face and sound, command drives the body.

### 5. Emotion-driven Expression and Sound Response
The robot's response to any input is conditioned on both the detected command and emotion simultaneously. The TFT renders six distinct eye geometries (drawn programmatically using primitive shapes — no image assets) with colour coding per emotion and a persistent idle blink animation. Sound selection uses a `(command, emotion)` lookup matrix — the same "dance" command produces a different sound when the speaker sounds happy versus sad. The entire response — face, motion sequence, and sound — is coordinated by a non-blocking state machine on the ESP32 that uses `millis()` timestamps throughout, with no blocking `delay()` calls in the main loop.

---

## Setup and Running

### Arduino (ESP32)

**Required libraries** — install via Arduino IDE Library Manager:
- ESP32Servo
- Adafruit GFX Library
- Adafruit ST7735 and ST7789 Library
- ArduinoJson (v7)

**Board settings:**

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| Partition Scheme | Huge APP (3MB No OTA 1MB SPIFFS) |
| PSRAM | OPI PSRAM |
| Flash Mode | QIO 80MHz |
| Upload Speed | 921600 |

Open `walle_esp32/walle_esp32.ino` and verify:
```cpp
#define PYTHON_SERVER_IP    "192.168.4.2"
#define PYTHON_SERVER_PORT  5001
```

Upload from the Arduino IDE. Note: uploading from macOS M1 may require manual boot mode entry (hold BOOT, press RESET, release BOOT) due to a stub flasher incompatibility with this chip revision. Windows upload works without any manual steps.

---

### Python Server

```bash
cd walle_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Download models** (requires internet, do this before connecting to ESP32 WiFi):
```bash
python -c "
from faster_whisper import WhisperModel
WhisperModel('tiny', device='cpu', compute_type='int8')
print('Whisper ready')
"

python -c "
from transformers import pipeline
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)
print('Emotion model ready')
"
```

**Generate placeholder sounds:**
```bash
python generate_sounds.py
```

**Train your own emotion classifier** (optional but recommended, ~25 min on M1):
```bash
python train_emotion_model.py
python train_fusion.py
```

**Run the server** (connect to `ESP32_TEST` WiFi first):
```bash
TRANSFORMERS_OFFLINE=1 python server.py
```

Verify the printed IP matches `PYTHON_SERVER_IP` in the sketch. Then press RESET on the ESP32 — the TFT should show "Python: OK".

**Web dashboard:** open `http://192.168.4.1` in a browser connected to `ESP32_TEST`.

---

## Voice Commands

Say the wake phrase first, then a command:

| Wake phrase | Command examples |
|---|---|
| "Hey buddy ..." | "... dance", "... wave", "... spin" |
| "Wake up ..." | "... go forward", "... move left" |

| Command | Trigger words | Action |
|---|---|---|
| dance | dance, groove, bounce | Head bob + arm alternation + motor pulses |
| wave | wave, hello, hi, hey | Right arm sweep x3 |
| spin | spin, rotate, turn around | Motor left spin |
| look | look, watch | Head sweep left-right-center |
| sleep | sleep, rest, tired | Servos droop slowly |
| stop | stop, halt, freeze | All servos to neutral, motors off |
| forward | go forward, walk, march | Motors forward 1.2s |
| backward | go back, reverse | Motors backward 1.2s |
| left | go left, move left | Motors left turn |
| right | go right, move right | Motors right turn |

---

## Configuration Reference

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `PYTHON_SERVER_IP` | `walle_esp32.ino` | `192.168.4.2` | Update if laptop gets different IP |
| `PYTHON_SERVER_PORT` | `walle_esp32.ino` | `5001` | Must match server PORT |
| `VAD_THRESHOLD` | `walle_esp32.ino` | `800` | Raise if false triggers, lower if mic too quiet |
| `RECORD_DURATION_MS` | `walle_esp32.ino` | `2500` | Recording window after VAD trigger |
| `WAKE_PHRASE_ACTIVE` | `server.py` | `True` | Set False to disable wake phrase requirement |
| `PORT` | `server.py` | `5001` | Flask server port |
| `MODEL_SIZE` | `stt.py` | `tiny` | Whisper model size (tiny/base/small) |

---

## Known Limitations

- **Acoustic fusion gap** — the multimodal fusion model was trained on simulated acoustic features (emotion-conditioned Gaussian distributions) due to the absence of a paired audio+label dataset. Real mic audio does not match the training distribution, so the fusion model is disabled at inference and text-only classification is used. A proper deployment would require collecting real labeled speech audio.
- **Power instability** — two 3.7V Li-Po cells in series (7.4V) cause brown-out resets when servos and motors draw simultaneous current spikes. A software brown-out disable is applied as a workaround. Proper fix is separate power rails for logic and actuators with bulk decoupling capacitors.
- **Speaker** — the MAX98357 I2S amplifier on this unit is non-functional. Audio output uses the laptop's speakers via pygame.

---

## Potential Future Work

- Replace VAD with `openwakeword` for true "Hey WALL-E" detection
- Collect real paired audio+label data and retrain the acoustic fusion model
- Personality drift — maintain a session-level mood state that influences response intensity
- WebSocket status push instead of 1-second polling in the dashboard
- Upgrade Whisper model from `tiny` to `base` for improved transcription accuracy
