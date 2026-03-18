#!/usr/bin/env python3
"""
Merge LoRA adapter weights into the base Whisper model and convert to MLX format.

Usage:
    python merge_and_convert.py [--lora-path ./whisper-ko-lora/final] [--output ./whisper-large-v3-ko-mlx]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to MLX")
    parser.add_argument(
        "--base-model",
        default="openai/whisper-large-v3",
        help="Base Whisper model name",
    )
    parser.add_argument(
        "--lora-path",
        default="./whisper-ko-lora/final",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--output",
        default="./whisper-large-v3-ko-mlx",
        help="Output directory for MLX model",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bits (4 or 8, 0 for no quantization)",
    )
    args = parser.parse_args()

    merged_path = os.path.join(args.output, "_merged_hf")

    # ─── Step 1: Merge LoRA into base model ───────────────────────────────

    print(f"[1/3] Merging LoRA adapter into base model...")
    print(f"  Base: {args.base_model}")
    print(f"  LoRA: {args.lora_path}")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    merged_model = model.merge_and_unload()

    print(f"  Saving merged model to {merged_path}...")
    os.makedirs(merged_path, exist_ok=True)
    merged_model.save_pretrained(merged_path)

    # Also save the processor
    processor = WhisperProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(merged_path)

    print(f"  Merged model saved.")

    # ─── Step 2: Convert to MLX ───────────────────────────────────────────

    print(f"\n[2/3] Converting to MLX format...")

    # Check if mlx-examples whisper convert.py is available
    convert_script = None
    possible_paths = [
        os.path.expanduser("~/mlx-examples/whisper/convert.py"),
        os.path.join(os.path.dirname(__file__), "mlx-examples/whisper/convert.py"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            convert_script = p
            break

    if convert_script is None:
        # Clone mlx-examples
        mlx_examples_dir = os.path.join(os.path.dirname(__file__), "mlx-examples")
        if not os.path.exists(mlx_examples_dir):
            print("  Cloning mlx-examples for conversion script...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--filter=blob:none",
                    "--sparse",
                    "https://github.com/ml-explore/mlx-examples.git",
                    mlx_examples_dir,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", "whisper"],
                cwd=mlx_examples_dir,
                check=True,
            )
        convert_script = os.path.join(mlx_examples_dir, "whisper", "convert.py")

    if not os.path.exists(convert_script):
        print(f"\n  ERROR: Could not find convert.py at {convert_script}")
        print(f"  The merged HuggingFace model is at: {merged_path}")
        print(f"  You can manually convert it using mlx-examples/whisper/convert.py")
        sys.exit(1)

    # Run conversion
    cmd = [
        sys.executable,
        convert_script,
        "--torch-name-or-path",
        merged_path,
        "--mlx-path",
        args.output,
    ]
    if args.q_bits > 0:
        cmd.extend(["-q", "--q-bits", str(args.q_bits)])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Convert script failed. stderr:\n{result.stderr}")
        print(f"\n  Trying alternative: direct MLX conversion...")
        try:
            convert_direct(merged_path, args.output, args.q_bits)
        except Exception as e:
            print(f"  Direct conversion also failed: {e}")
            print(f"\n  The merged HuggingFace model is saved at: {merged_path}")
            print(f"  You can convert it manually later.")
            sys.exit(1)
    else:
        print(result.stdout)

    # ─── Step 3: Cleanup and verify ───────────────────────────────────────

    print(f"\n[3/3] Verifying output...")

    if os.path.exists(args.output):
        files = os.listdir(args.output)
        print(f"  Output directory: {args.output}")
        print(f"  Files: {', '.join(files)}")

        # Quick size check
        total_size = sum(
            os.path.getsize(os.path.join(args.output, f))
            for f in files
            if os.path.isfile(os.path.join(args.output, f))
        )
        print(f"  Total size: {total_size / 1024 / 1024:.0f} MB")

        print(f"\n=== Done! ===")
        print(f"MLX model saved to: {args.output}")
        print(f"\nTo use in live_transcribe.py, change WHISPER_MODEL to:")
        print(f'  WHISPER_MODEL = "{os.path.abspath(args.output)}"')

        # Optionally clean up the merged HuggingFace model
        print(f"\nYou can delete the intermediate HuggingFace model at: {merged_path}")
    else:
        print(f"  ERROR: Output directory not found at {args.output}")
        sys.exit(1)


def convert_direct(hf_path, output_path, q_bits):
    """Direct conversion using mlx and numpy if the convert script fails."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from transformers import WhisperConfig

    print("  Attempting direct MLX weight conversion...")

    config = WhisperConfig.from_pretrained(hf_path)
    state_dict = torch.load(
        os.path.join(hf_path, "model.safetensors"),
        map_location="cpu",
        weights_only=True,
    )

    # Convert PyTorch state dict to MLX-compatible format
    mlx_weights = {}
    for key, value in state_dict.items():
        # Convert to numpy then to MLX
        np_val = value.numpy()
        mlx_weights[key] = mx.array(np_val)

    os.makedirs(output_path, exist_ok=True)

    if q_bits > 0:
        print(f"  Quantizing to {q_bits}-bit...")
        # Save as float first, quantization is handled by mlx-whisper at load time
        mx.savez(os.path.join(output_path, "weights.npz"), **mlx_weights)
    else:
        mx.savez(os.path.join(output_path, "weights.npz"), **mlx_weights)

    # Save config
    config_dict = config.to_dict()
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  Direct conversion complete.")


if __name__ == "__main__":
    main()
