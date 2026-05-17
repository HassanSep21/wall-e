"""
nlp.py — NLP Pipeline (V2 — Multimodal)
  - M10a: Your fine-tuned DistilBERT text emotion classifier
  - M10b: Multimodal fusion MLP (text + acoustic features)
  - M10c: Command rule engine (regex keyword matching)

Falls back gracefully:
  - If fusion model exists → use multimodal (text + acoustic)
  - If only text model exists → use text-only
  - If neither exists → use HuggingFace pretrained (original behaviour)
"""

import re
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
TEXT_MODEL_DIR  = Path("models/text_emotion")
FUSION_MODEL    = Path("models/fusion_mlp.pt")
FUSION_CONFIG   = Path("models/fusion_config.json")
LABELS_FILE     = Path("models/text_emotion_labels.json")

# ── Fallback HuggingFace model (used only if no local model found) ───────────
HF_FALLBACK_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ── Config ───────────────────────────────────────────────────────────────────
MIN_EMOTION_CONF  = 0.40
MIN_STT_CONF      = 0.45
N_MFCC            = 13

# ── Internal labels ──────────────────────────────────────────────────────────
INTERNAL_LABELS = ["happy", "sad", "angry", "surprised", "curious", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(INTERNAL_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTERNAL_LABELS)}

# HuggingFace fallback label mapping
HF_TO_INTERNAL = {
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "angry",
    "fear":     "sad",
    "surprise": "surprised",
    "neutral":  "neutral",
    "disgust":  "angry",
}

# ── Command rules ─────────────────────────────────────────────────────────────
COMMAND_RULES: dict[str, list[str]] = {
    # Movement commands MUST come before dance — more specific patterns first
    "forward":  [r"\bgo forward\b", r"\bmove forward\b", r"\bgo ahead\b",
                 r"\bwalk\b", r"\bmarch\b", r"\bforward\b"],
    "backward": [r"\bgo back\b", r"\bmove back\b", r"\bbackward\b",
                 r"\bback up\b", r"\breverse\b", r"\bretreat\b"],
    "left":     [r"\bgo left\b", r"\bmove left\b", r"\bturn left\b"],
    "right":    [r"\bgo right\b", r"\bmove right\b", r"\bturn right\b"],

    # Dance — remove r"\bmov(e|ing)\b" which was stealing movement commands
    "dance":    [r"\bdanc", r"\bgroov", r"\bbounc", r"\bjig\b", r"\bshuffle\b"],

    "wave":     [r"\bwave", r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreet"],
    "stop":     [r"\bstop\b", r"\bno\b", r"\bdon.?t\b", r"\bquit\b",
                 r"\bhalt\b", r"\bfreeze\b"],
    "spin":     [r"\bspin\b", r"\bturn around\b", r"\brotat", r"\bcircle"],
    "look":     [r"\blook\b", r"\bwatch\b", r"\bsee\b"],
    "sleep":    [r"\bsleep\b", r"\brest\b", r"\bnap\b", r"\btired\b", r"\bnight\b"],
}


class NLPPipeline:
    """
    Multimodal NLP pipeline.
    Auto-detects which models are available and uses the best one.

    Usage:
        pipeline = NLPPipeline()
        pipeline.load()
        result = pipeline.process(transcript, stt_confidence, pcm_bytes)
    """

    def __init__(self):
        self._mode           = None   # "fusion" | "text_local" | "hf_fallback"
        self._tokenizer      = None
        self._text_model     = None
        self._fusion_model   = None
        self._fusion_config  = None
        self._hf_classifier  = None
        self._device         = None

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self):
        import torch
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Priority 1: fusion model (multimodal)
        if FUSION_MODEL.exists() and FUSION_CONFIG.exists() and TEXT_MODEL_DIR.exists():
            self._load_fusion()
            self._mode = "fusion"
            logger.info("NLP mode: MULTIMODAL FUSION (text + acoustic)")

        # Priority 2: local text model only
        elif TEXT_MODEL_DIR.exists():
            self._load_text_model()
            self._mode = "text_local"
            logger.info("NLP mode: LOCAL TEXT MODEL (fine-tuned DistilBERT)")

        # Priority 3: HuggingFace fallback
        else:
            self._load_hf_fallback()
            self._mode = "hf_fallback"
            logger.info(f"NLP mode: HF FALLBACK ({HF_FALLBACK_MODEL})")

        logger.info(f"NLP pipeline ready (mode={self._mode}, device={self._device})")

    def _load_text_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info(f"Loading local text model from {TEXT_MODEL_DIR}...")
        t0 = time.time()
        self._tokenizer   = AutoTokenizer.from_pretrained(str(TEXT_MODEL_DIR))
        self._text_model  = AutoModelForSequenceClassification.from_pretrained(
            str(TEXT_MODEL_DIR)
        ).to(self._device)
        self._text_model.eval()
        logger.info(f"Local text model loaded in {time.time()-t0:.2f}s")

    def _load_fusion(self):
        import torch
        from train_fusion import FusionMLP

        # Load text model first
        self._load_text_model()

        # Load fusion config
        with open(FUSION_CONFIG) as f:
            self._fusion_config = json.load(f)

        # Load fusion MLP
        logger.info(f"Loading fusion MLP from {FUSION_MODEL}...")
        self._fusion_model = FusionMLP(
            text_dim     = self._fusion_config["text_embed_dim"],
            acoustic_dim = self._fusion_config["acoustic_dim"],
            hidden1      = self._fusion_config["fusion_hidden1"],
            hidden2      = self._fusion_config["fusion_hidden2"],
            num_classes  = self._fusion_config["num_classes"],
        ).to(self._device)
        self._fusion_model.load_state_dict(
            torch.load(str(FUSION_MODEL), map_location=self._device)
        )
        self._fusion_model.eval()
        logger.info("Fusion MLP loaded")

    def _load_hf_fallback(self):
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading HF fallback model [{HF_FALLBACK_MODEL}]...")
        t0 = time.time()
        self._hf_classifier = hf_pipeline(
            "text-classification",
            model=HF_FALLBACK_MODEL,
            top_k=None,
            device=-1,
        )
        logger.info(f"HF model loaded in {time.time()-t0:.2f}s")

    # ── Process ──────────────────────────────────────────────────────────────

    def process(self,
                transcript:     str,
                stt_confidence: float = 1.0,
                pcm_bytes:      Optional[bytes] = None) -> dict:
        """
        Run full NLP pipeline.

        Args:
            transcript:      STT transcript text
            stt_confidence:  STT confidence [0,1]
            pcm_bytes:       Raw 16-bit PCM audio bytes (for acoustic features)
                             If None or mode is not fusion, acoustic features are skipped.

        Returns:
            {emotion, command, emotion_confidence, command_matched}
        """
        if not transcript or stt_confidence < MIN_STT_CONF:
            return _default_result()

        t0 = time.time()

        if False and self._mode == "fusion" and pcm_bytes:
            emotion, conf = self._classify_multimodal(transcript, pcm_bytes)
        elif self._mode in ("fusion", "text_local"):
            emotion, conf = self._classify_text_local(transcript)
        else:
            emotion, conf = self._classify_hf(transcript)

        command, matched = _match_command(transcript)
        
        # Short transcript with low emotion confidence — use sensible defaults
        if len(transcript.split()) <= 2 and conf < 0.55:
            if command != "idle":
                emotion = "happy"
                conf = 0.60
            else:
                emotion = "neutral"
                conf = 0.60
                
        elapsed_ms = int((time.time() - t0) * 1000)

        
        actual_mode = "text_local" if self._mode == "fusion" else self._mode
        logger.info(
            f"NLP [{elapsed_ms}ms] [{actual_mode}] "
            f"emotion={emotion}({conf:.2f}) command={command} | \"{transcript}\""
        )

        return {
            "emotion":            emotion,
            "command":            command,
            "emotion_confidence": conf,
            "command_matched":    matched,
        }

    # ── Classifiers ──────────────────────────────────────────────────────────

    def _classify_multimodal(self, text: str, pcm_bytes: bytes) -> tuple[str, float]:
        """Fusion: text embedding + acoustic features → emotion."""
        import torch
        try:
            # Text embedding (CLS token)
            text_emb = self._get_text_embedding(text)   # (1, 768)

            # Acoustic features from raw PCM
            acoustic = _extract_acoustic(pcm_bytes)      # (1, 29)

            # Normalize using training stats
            ac_mean = np.array(self._fusion_config["acoustic_mean"], dtype=np.float32)
            ac_std  = np.array(self._fusion_config["acoustic_std"],  dtype=np.float32)
            te_mean = np.array(self._fusion_config["text_mean"],     dtype=np.float32)
            te_std  = np.array(self._fusion_config["text_std"],      dtype=np.float32)

            text_emb  = (text_emb  - te_mean) / (te_std  + 1e-8)
            acoustic   = (acoustic  - ac_mean) / (ac_std  + 1e-8)

            text_t  = torch.tensor(text_emb,  dtype=torch.float32).to(self._device)
            acou_t  = torch.tensor(acoustic,  dtype=torch.float32).to(self._device)

            with torch.no_grad():
                logits = self._fusion_model(text_t, acou_t)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            top_id  = int(probs.argmax())
            conf    = float(probs[top_id])
            emotion = ID2LABEL.get(top_id, "neutral")

            if conf < MIN_EMOTION_CONF:
                return "neutral", conf
            return emotion, round(conf, 3)

        except Exception as e:
            logger.error(f"Multimodal classify error: {e}, falling back to text")
            return self._classify_text_local(text)

    def _classify_text_local(self, text: str) -> tuple[str, float]:
        """Local fine-tuned DistilBERT text classification."""
        import torch
        try:
            encoded = self._tokenizer(
                text[:512],
                truncation=True,
                padding=True,
                max_length=64,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._text_model(**encoded).logits
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            top_id  = int(probs.argmax())
            conf    = float(probs[top_id])

            # Map using model's own id2label if available
            id2label = self._text_model.config.id2label
            emotion  = id2label.get(top_id, ID2LABEL.get(top_id, "neutral"))

            # Ensure it's in our internal set
            if emotion not in INTERNAL_LABELS:
                emotion = HF_TO_INTERNAL.get(emotion, "neutral")

            if conf < MIN_EMOTION_CONF:
                return "neutral", conf
            return emotion, round(conf, 3)

        except Exception as e:
            logger.error(f"Local text classify error: {e}")
            return "neutral", 0.0

    def _classify_hf(self, text: str) -> tuple[str, float]:
        """HuggingFace fallback classifier."""
        try:
            results = self._hf_classifier(text[:512])
            if not results or not results[0]:
                return "neutral", 0.0
            scores  = sorted(results[0], key=lambda x: x["score"], reverse=True)
            top     = scores[0]
            label   = HF_TO_INTERNAL.get(top["label"].lower(), "neutral")
            conf    = round(float(top["score"]), 3)
            if conf < MIN_EMOTION_CONF:
                return "neutral", conf
            return label, conf
        except Exception as e:
            logger.error(f"HF classify error: {e}")
            return "neutral", 0.0

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get CLS embedding from fine-tuned DistilBERT. Returns (1, 768)."""
        import torch
        encoded = self._tokenizer(
            text, truncation=True, padding=True,
            max_length=64, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._text_model.distilbert(**encoded)
            cls = out.last_hidden_state[:, 0, :]
        return cls.cpu().numpy()   # (1, 768)


# ── Acoustic feature extraction (live inference) ──────────────────────────────

def _extract_acoustic(pcm_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Extract acoustic features from raw 16-bit PCM bytes.
    Returns shape (1, ACOUSTIC_DIM) = (1, 29).
    Safe — returns zeros on any error.
    """
    try:
        import librosa
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < 512 or np.max(np.abs(audio)) < 1e-6:
            return np.zeros((1, N_MFCC * 2 + 3), dtype=np.float32)

        # MFCCs
        mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std  = mfcc.std(axis=1)

        # Pitch
        try:
            f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr, frame_length=512)
            pitch_mean = float(np.nanmean(f0)) if np.any(voiced) else 0.0
        except Exception:
            pitch_mean = 0.0

        # Energy + ZCR
        energy_mean = float(librosa.feature.rms(y=audio).mean())
        zcr_mean    = float(librosa.feature.zero_crossing_rate(audio).mean())

        features = np.concatenate([
            mfcc_mean, mfcc_std, [pitch_mean, energy_mean, zcr_mean]
        ]).astype(np.float32)

        return features.reshape(1, -1)   # (1, 29)

    except Exception as e:
        logger.error(f"Acoustic feature extraction error: {e}")
        return np.zeros((1, N_MFCC * 2 + 3), dtype=np.float32)


# ── Command rule engine ───────────────────────────────────────────────────────

def _match_command(transcript: str) -> tuple[str, bool]:
    text = transcript.lower().strip()
    for command, patterns in COMMAND_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return command, True
    return "idle", False


def _default_result() -> dict:
    return {
        "emotion":            "neutral",
        "command":            "idle",
        "emotion_confidence": 0.0,
        "command_matched":    False,
    }
    