#!/usr/bin/env python3
"""
Fine-tune Whisper large-v3 for Korean using LoRA (PEFT).

Uses Mozilla Common Voice Korean dataset via datacollective (Mozilla Data Collective).
Designed for Apple Silicon (MPS backend) with 24GB unified memory.

Usage:
    python finetune_whisper_ko.py [--model openai/whisper-large-v3] [--epochs 3] [--resume]

Requires:
    MDC_API_KEY environment variable for datacollective dataset download.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import Audio, Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "./whisper-ko-lora"
MDC_DATASET_ID = "mcv-scripted-ko-v24.0"
LANGUAGE = "korean"
TASK = "transcribe"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator that pads inputs and labels separately."""

    processor: Any

    def __call__(self, features):
        # Split inputs and labels since they have different padding requirements
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Tokenized labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if the model prepends it automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, feature_extractor, tokenizer):
    """Process a single example: extract features and tokenize."""
    audio = batch["audio"]

    # Compute log-Mel spectrogram
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Tokenize transcription
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred, tokenizer, metric):
    """Compute CER on predictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Korean")
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Base Whisper model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples (for testing the pipeline)",
    )
    args = parser.parse_args()

    print(f"=== Fine-tuning {args.model} for Korean ===")
    print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print(f"Output: {OUTPUT_DIR}")

    # ─── Load processor components ────────────────────────────────────────

    print("\n[1/6] Loading processor...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        args.model, language=LANGUAGE, task=TASK
    )

    # ─── Load dataset ─────────────────────────────────────────────────────

    print("\n[2/6] Loading Common Voice Korean dataset via datacollective...")

    if not os.environ.get("MDC_API_KEY"):
        print("  ERROR: MDC_API_KEY environment variable is required.")
        print("  Set it with: export MDC_API_KEY=<your-key>")
        sys.exit(1)

    import pandas as pd
    import datacollective

    # Download the archive (skips if already present)
    datacollective.download_dataset(MDC_DATASET_ID)

    # datacollective.load_dataset() has a schema bug for this dataset,
    # so we load the TSVs directly from the extracted archive.
    from datacollective.download import resolve_download_dir

    base_dir = resolve_download_dir(None)
    extract_dir = base_dir / MDC_DATASET_ID

    # Find the language directory inside the extracted corpus
    ko_dirs = list(extract_dir.rglob("ko"))
    if not ko_dirs:
        print(f"  ERROR: Could not find 'ko' directory in {extract_dir}")
        sys.exit(1)
    ko_dir = ko_dirs[0]
    clips_dir = ko_dir / "clips"
    print(f"  Data directory: {ko_dir}")
    print(f"  Clips: {len(list(clips_dir.iterdir()))} MP3 files")

    # Load train+dev and test splits from TSV files
    train_df = pd.read_csv(ko_dir / "train.tsv", sep="\t")
    dev_df = pd.read_csv(ko_dir / "dev.tsv", sep="\t")
    test_df = pd.read_csv(ko_dir / "test.tsv", sep="\t")
    train_df = pd.concat([train_df, dev_df], ignore_index=True)

    # Build full audio paths
    train_df["audio"] = train_df["path"].apply(lambda p: str(clips_dir / p))
    test_df["audio"] = test_df["path"].apply(lambda p: str(clips_dir / p))

    # Convert to HuggingFace Datasets
    common_voice = DatasetDict()
    common_voice["train"] = Dataset.from_dict({
        "audio": train_df["audio"].tolist(),
        "sentence": train_df["sentence"].tolist(),
    })
    common_voice["test"] = Dataset.from_dict({
        "audio": test_df["audio"].tolist(),
        "sentence": test_df["sentence"].tolist(),
    })

    print(f"  Train: {len(common_voice['train'])} samples")
    print(f"  Test:  {len(common_voice['test'])} samples")

    if args.max_train_samples:
        common_voice["train"] = common_voice["train"].select(
            range(min(args.max_train_samples, len(common_voice["train"])))
        )
        common_voice["test"] = common_voice["test"].select(
            range(min(args.max_train_samples // 5, len(common_voice["test"])))
        )
        print(f"  (Limited to {len(common_voice['train'])} train, {len(common_voice['test'])} test)")

    # Cast audio column to load MP3 files and resample to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # ─── Process dataset ──────────────────────────────────────────────────

    print("\n[3/6] Processing dataset (feature extraction + tokenization)...")
    common_voice = common_voice.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=["audio", "sentence"],
        num_proc=1,  # MPS doesn't benefit from multiprocess here
    )

    # Filter out samples with labels exceeding Whisper's max target length (448 tokens)
    max_label_length = 448
    before_train = len(common_voice["train"])
    before_test = len(common_voice["test"])
    common_voice["train"] = common_voice["train"].filter(
        lambda x: len(x["labels"]) <= max_label_length
    )
    common_voice["test"] = common_voice["test"].filter(
        lambda x: len(x["labels"]) <= max_label_length
    )
    print(f"  Filtered: {before_train} -> {len(common_voice['train'])} train, "
          f"{before_test} -> {len(common_voice['test'])} test (max {max_label_length} label tokens)")

    # ─── Load model with LoRA ─────────────────────────────────────────────

    print("\n[4/6] Loading model and applying LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = False  # Required for gradient checkpointing
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── Metrics ──────────────────────────────────────────────────────────

    metric = evaluate.load("cer")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ─── Training ─────────────────────────────────────────────────────────

    print("\n[5/6] Starting training...")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size = 16
        learning_rate=1e-3,
        num_train_epochs=args.epochs,
        warmup_steps=50,
        fp16=False,
        bf16=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=25,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,  # Avoid multiprocessing issues on macOS
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric),
        processing_class=processor.feature_extractor,
    )

    # Run training
    if args.resume and os.path.exists(OUTPUT_DIR):
        print("  Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ─── Save ─────────────────────────────────────────────────────────────

    print("\n[6/6] Saving LoRA adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    # Final evaluation (loss only, no generation to save memory)
    print("\n=== Final Evaluation ===")
    results = trainer.evaluate()
    print(f"  Eval loss: {results['eval_loss']:.4f}")
    print(f"\nDone! LoRA adapter saved to {OUTPUT_DIR}/final")
    print(f"Next step: python merge_and_convert.py")


if __name__ == "__main__":
    main()
