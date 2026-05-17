"""
train_emotion_model.py
Fine-tunes DistilBERT on the dair-ai/emotion dataset (20k samples, 6 classes).
Run once on normal WiFi. Saves model locally for offline use.

Usage:
    python train_emotion_model.py

Output:
    models/text_emotion/         ← your trained model
    models/text_emotion_labels.json
"""

import os
import json
import time
import numpy as np
from pathlib import Path

# ── Output paths ────────────────────────────────────────────────────────────
MODEL_OUT_DIR  = Path("models/text_emotion")
LABELS_FILE    = Path("models/text_emotion_labels.json")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Training config ──────────────────────────────────────────────────────────
BASE_MODEL     = "distilbert-base-uncased"
NUM_EPOCHS     = 3
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
MAX_LENGTH     = 64       # max token length — emotion sentences are short
SEED           = 42

# ── dair-ai/emotion label map ────────────────────────────────────────────────
# Dataset labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
# We map these to our internal emotion set
DATASET_TO_INTERNAL = {
    0: "sad",
    1: "happy",
    2: "happy",      # love → happy for V1
    3: "angry",
    4: "sad",        # fear → sad for V1
    5: "surprised",
}

# Our internal labels → index
INTERNAL_LABELS = ["happy", "sad", "angry", "surprised", "curious", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(INTERNAL_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTERNAL_LABELS)}


def main():
    print("=" * 60)
    print("  WALL-E Emotion Classifier — Fine-tuning DistilBERT")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n[1/5] Loading dair-ai/emotion dataset...")
    from datasets import load_dataset
    raw = load_dataset("dair-ai/emotion")
    print(f"      Train: {len(raw['train'])} samples")
    print(f"      Val:   {len(raw['validation'])} samples")
    print(f"      Test:  {len(raw['test'])} samples")

    # ── 2. Remap labels to our internal set ─────────────────────────────────
    print("\n[2/5] Remapping labels to internal emotion set...")

    def remap(example):
        internal = DATASET_TO_INTERNAL.get(example["label"], "neutral")
        example["label"] = LABEL2ID[internal]
        return example

    dataset = raw.map(remap)

    # Print class distribution
    from collections import Counter
    train_labels = [ex["label"] for ex in dataset["train"]]
    dist = Counter(train_labels)
    print("      Class distribution (train):")
    for lid, count in sorted(dist.items()):
        print(f"        {ID2LABEL[lid]:<12} {count:>5} samples")

    # ── 3. Tokenize ──────────────────────────────────────────────────────────
    print(f"\n[3/5] Tokenizing with {BASE_MODEL}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True, batch_size=256)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    print("      ✓ Tokenization complete")

    # ── 4. Load model + fine-tune ────────────────────────────────────────────
    print(f"\n[4/5] Loading {BASE_MODEL} and fine-tuning...")
    print(f"      Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"      This will take ~20-35 minutes on M1 CPU...\n")

    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import torch

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTERNAL_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Use MPS (Apple Silicon GPU) if available for faster training
    if torch.backends.mps.is_available():
        print("      ✓ Apple Silicon MPS detected — using GPU acceleration")
        device = "mps"
    else:
        print("      ℹ Running on CPU")
        device = "cpu"

    training_args = TrainingArguments(
        output_dir="models/text_emotion_checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=SEED,
        report_to="none",
        disable_tqdm=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        # Per-class accuracy
        per_class = {}
        for cid, cname in ID2LABEL.items():
            mask = labels == cid
            if mask.sum() > 0:
                per_class[cname] = (predictions[mask] == labels[mask]).mean()
        return {"accuracy": float(accuracy), **{f"acc_{k}": float(v)
                                                 for k, v in per_class.items()}}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n      ✓ Training complete in {elapsed/60:.1f} minutes")

    # ── 5. Evaluate + save ───────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set and saving model...")

    results = trainer.evaluate(tokenized["test"])
    print(f"\n      Test accuracy : {results['eval_accuracy']*100:.1f}%")
    for key, val in results.items():
        if key.startswith("eval_acc_"):
            emotion = key.replace("eval_acc_", "")
            print(f"        {emotion:<12} {val*100:.1f}%")

    # Save model + tokenizer
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))

    # Save label map
    label_meta = {
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "internal_labels": INTERNAL_LABELS,
        "base_model": BASE_MODEL,
        "test_accuracy": results["eval_accuracy"],
        "trained_on": "dair-ai/emotion",
    }
    with open(LABELS_FILE, "w") as f:
        json.dump(label_meta, f, indent=2)

    print(f"\n      ✓ Model saved to: {MODEL_OUT_DIR}")
    print(f"      ✓ Labels saved to: {LABELS_FILE}")
    print()
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Test accuracy: {results['eval_accuracy']*100:.1f}%")
    print(f"  Model size: {_dir_size_mb(MODEL_OUT_DIR):.0f} MB")
    print()
    print("  Next step: python train_fusion.py")
    print("=" * 60)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


if __name__ == "__main__":
    main()
    