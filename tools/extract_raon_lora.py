"""Salvage a peft LoRA adapter from an old RaonDuplexModel Trainer checkpoint.

Background
----------
Our training setup wraps ``model.text_model`` (Qwen3 backbone) with peft but
leaves the outer ``RaonDuplexModel`` un-wrapped. HF Trainer therefore saves a
plain ``model.safetensors`` containing the full outer state_dict — no
``adapter_config.json`` or ``adapter_model.safetensors`` — which makes
``PeftModel.from_pretrained`` refuse to load the checkpoint.

This one-shot utility reads the saved full model, filters out the peft
sub-tree (``text_model.base_model.model.*.lora_[AB].*``), strips the
``text_model.`` prefix, and writes a proper adapter directory that
``PeftModel.from_pretrained`` can consume.

Usage
-----
::

    uv run python tools/extract_raon_lora.py \\
        --checkpoint_dir ../saves/raon/gametime_lora/checkpoint-2250 \\
        --output_dir     ../saves/raon/gametime_lora/checkpoint-2250/lora_adapter \\
        --lora_rank 128 --lora_alpha 128

The ``--lora_*`` args MUST match what you trained with (see train_raon.sh).
For runs going forward the trainer writes this folder automatically.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger("extract_raon_lora")


def extract(
    checkpoint_dir: Path,
    output_dir: Path,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
) -> None:
    shards = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not shards:
        raise SystemExit(f"No model*.safetensors found under {checkpoint_dir}")

    prefix = "text_model."
    extracted: dict[str, torch.Tensor] = {}
    total_keys = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                total_keys += 1
                if not key.startswith(prefix):
                    continue
                inner = key[len(prefix):]
                # LoRA-only: skip frozen base_layer weights (peft has them
                # via base model weights already when you re-apply).
                if "lora_A" not in inner and "lora_B" not in inner:
                    continue
                # Mirror peft.get_peft_model_state_dict: strip ".default" so
                # that PeftModel.from_pretrained's rename step maps
                # `lora_A.weight` -> `lora_A.default.weight` cleanly instead
                # of double-inserting to `.default.default.weight`.
                inner = inner.replace(".default", "")
                extracted[inner] = f.get_tensor(key)

    if not extracted:
        raise SystemExit(
            f"Scanned {total_keys} keys across {len(shards)} shard(s) but found no "
            f"lora_A/lora_B weights under '{prefix}*'. Either this isn't a LoRA run, "
            f"or the wrapping convention changed."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter_model.safetensors"
    save_file(extracted, str(adapter_path))
    logger.info("Wrote %d tensors -> %s (%.1f MB)",
                len(extracted), adapter_path,
                adapter_path.stat().st_size / 1e6)

    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=None,
    )
    cfg.save_pretrained(str(output_dir))
    logger.info("Wrote adapter_config.json -> %s", output_dir / "adapter_config.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", type=Path, required=True,
                   help="HF Trainer output dir (contains model*.safetensors + trainer_state.json).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write adapter_config.json + adapter_model.safetensors.")
    p.add_argument("--lora_rank", type=int, default=128,
                   help="Must match the --lora_rank used during training.")
    p.add_argument("--lora_alpha", type=int, default=128,
                   help="Must match the --lora_alpha used during training.")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = p.parse_args()

    extract(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )


if __name__ == "__main__":
    main()
