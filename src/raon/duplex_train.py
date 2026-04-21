# Copyright 2026 The RAON Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full-duplex fine-tuning script for RAON.

Usage:
    python -m raon.duplex_train \
        --model_path /path/to/duplex/model \
        --data_dir /path/to/data_or_jsonl \
        --output_dir /path/to/output \
        --max_steps 100 \
        --learning_rate 1e-5
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainerCallback, TrainingArguments


from raon.utils.training_callbacks import StepLoggingCallback, SaveTokenizerCallback


from raon.models.raon import RaonDuplexModel
from raon.utils.duplex_data import duplex_collate_fn, make_raon_duplex_data_module
from raon.utils.processor import RaonProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RAON duplex model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Data path(s) for training. Accepts: "
            "a single jsonl, comma-separated jsonl files, "
            "a single directory, or comma-separated directories. "
            "Directories are expanded to all *.jsonl files inside."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["fa", "sdpa", "eager"],
        help="Attention implementation (default: sdpa). Use `fa` for FlashAttention.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help=(
            "Max token sequence length for per-channel filtering/truncation. "
            "Overlength channels are filtered; if all are overlength, the shortest one is truncated. "
            "(default: 8192)"
        ),
    )
    parser.add_argument(
        "--use_packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sequence packing for duplex training (default: enabled). Use --no-use_packing to disable.",
    )
    parser.add_argument(
        "--max_packed_seq_length",
        type=int,
        default=8192,
        help="Maximum packed sequence length in tokens (default: 8192).",
    )
    parser.add_argument(
        "--max_audio_seq_length",
        type=int,
        default=192000,
        help="Maximum audio chunk length in samples for packed collation (default: 192000).",
    )
    # Loss weights
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--audio_output_pad_text_loss_weight", type=float, default=0.0)
    parser.add_argument("--audio_end_text_loss_weight", type=float, default=1.0)
    parser.add_argument("--epad_loss_weight", type=float, default=1.0)
    parser.add_argument("--sil_loss_weight", type=float, default=1.0)
    parser.add_argument("--bc_loss_weight", type=float, default=1.0)
    parser.add_argument("--semantic_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--acoustic_loss_weight",
        type=float,
        default=0.1,
        help="Loss weight applied to all acoustic codebooks (default: 0.1).",
    )
    # LoRA (optional; text_model backbone only)
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, wrap model.text_model with peft.LoraConfig so only LoRA adapters are trainable.",
    )
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA (default covers Qwen-family attn+MLP).",
    )
    # Experiment tracking (wandb / tensorboard). HF Trainer native; 'none' disables.
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Comma-separated HF Trainer integrations: none | all | wandb | tensorboard | ...",
    )
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Sets WANDB_PROJECT env var if provided.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Sets WANDB_NAME env var if provided.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.attn_implementation == "fa":
        args.attn_implementation = "flash_attention_2"
    if args.batch_size != 1:
        logger.warning("duplex_train enforces batch_size=1. Overriding --batch_size=%s to 1.", args.batch_size)
    args.batch_size = 1

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from raon.utils.misc import resolve_dtype

    torch_dtype = resolve_dtype(args.dtype)

    # Load model
    logger.info("Loading model from %s ...", args.model_path)
    model = RaonDuplexModel.from_pretrained(args.model_path, torch_dtype=torch_dtype)
    model._set_attention_implementation(args.attn_implementation)

    # Freeze modules: audio_encoder + aligner modules frozen, LLM backbone trainable.
    _frozen_modules = [
        "audio_encoder", "input_adaptor", "output_adaptor",
        "audio_lm_head", "proj_code", "code_predictor", "speaker_encoder",
    ]
    for module_name in _frozen_modules:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
            logger.info("Froze %s", module_name)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        for p in model.text_model.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            bias="none",
            task_type=None,
        )
        model.text_model = get_peft_model(model.text_model, lora_cfg)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA on text_model: rank=%d alpha=%d trainable=%.2fM / total=%.2fM (%.3f%%)",
            args.lora_rank, args.lora_alpha,
            n_trainable / 1e6, n_total / 1e6, 100.0 * n_trainable / max(n_total, 1),
        )

    # Loss weights (from args)
    model.text_loss_weight = args.text_loss_weight
    model.audio_output_pad_text_loss_weight = args.audio_output_pad_text_loss_weight
    model.audio_end_text_loss_weight = args.audio_end_text_loss_weight
    model.epad_loss_weight = args.epad_loss_weight
    model.sil_loss_weight = args.sil_loss_weight
    model.bc_loss_weight = args.bc_loss_weight
    model.semantic_loss_weight = args.semantic_loss_weight
    model.acoustic_loss_weights = [args.acoustic_loss_weight] * model.num_code_groups

    # Load processor
    logger.info("Loading processor from %s ...", args.model_path)
    processor = RaonProcessor.from_pretrained(args.model_path)

    # Build data module
    from raon.utils.data import resolve_data_dir

    jsonl_paths = resolve_data_dir(args.data_dir)
    data_module = make_raon_duplex_data_module(
        processor=processor,
        jsonl_paths=jsonl_paths,
    )

    # wrap collator with max_seq_length filter
    max_seq_length = args.max_seq_length

    # Ported from duplex-model/duplex_dataset/util.py: update_audio_lengths_single
    fr = processor.frame_rate
    sr = processor.sampling_rate
    from raon.utils.special_tokens import (
        AUDIO_INPUT_PLACEHOLDER,
        AUDIO_OUTPUT_PLACEHOLDER,
        LOSS_IGNORE_INDEX,
        SPEAKER_EMBEDDING_PLACEHOLDER,
    )

    def _update_audio_lengths_single(
        input_ids: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_token_id: int,
    ) -> torch.Tensor | None:
        """Ported from duplex-model: clamp audio lengths to packed placeholder budget."""
        if audio_lengths is None:
            return None

        assert input_ids.shape[0] == 1, (
            f"_update_audio_lengths_single: Expected batch size 1 but got `{input_ids.shape[0]=}`."
        )

        num_audio_tokens_from_input_ids = (input_ids[:, :max_seq_length] == audio_token_id).long().sum(dim=1)
        num_audio_tokens_from_audio_lengths = (audio_lengths * fr / sr).ceil().long()
        if num_audio_tokens_from_audio_lengths.sum() < num_audio_tokens_from_input_ids.sum():
            logger.warning(
                "_update_audio_lengths_single: Not enough audio for audio tokens in input_ids. "
                "Available: %d, required: %d. Clamping audio lengths to available budget.",
                num_audio_tokens_from_audio_lengths.sum().item(),
                num_audio_tokens_from_input_ids.sum().item(),
            )
            num_audio_tokens_from_input_ids = num_audio_tokens_from_input_ids.clamp(
                max=num_audio_tokens_from_audio_lengths.sum()
            )
        total_num_audio_tokens = num_audio_tokens_from_audio_lengths.cumsum(dim=0).minimum(num_audio_tokens_from_input_ids)
        num_audio_tokens_from_audio_lengths = torch.cat(
            (total_num_audio_tokens[:1], total_num_audio_tokens[1:] - total_num_audio_tokens[:-1])
        )
        max_audio_length = (num_audio_tokens_from_audio_lengths.double() * sr / fr).long()
        return audio_lengths.minimum(max_audio_length)

    def _fix_duplex_labels(channel: dict[str, Any]) -> dict[str, Any]:
        """Match duplex-model fix_duplex_labels: mask everything after last [A] token."""
        result = dict(channel)
        input_ids = result["input_ids"]
        assert input_ids.shape[0] == 1, f"_fix_duplex_labels expects batch size 1, got {input_ids.shape[0]}"
        audio_output_positions = (input_ids[0] == AUDIO_OUTPUT_PLACEHOLDER.id).nonzero(as_tuple=True)[0]
        if audio_output_positions.numel() == 0:
            return result

        last_valid_index = int(audio_output_positions[-1].item())
        if last_valid_index + 1 >= input_ids.shape[1]:
            return result

        result["input_ids"] = input_ids.clone()
        result["attention_mask"] = result["attention_mask"].clone()
        result["labels"] = result["labels"].clone()
        result["input_ids"][:, last_valid_index + 1 :] = 0
        result["attention_mask"][:, last_valid_index + 1 :] = 0
        result["labels"][:, last_valid_index + 1 :] = LOSS_IGNORE_INDEX
        return result

    def _pad_truncate_channel(channel: dict[str, Any]) -> dict[str, Any]:
        """Truncate sequence fields, apply duplex label fix, and synchronize audio lengths."""
        result = dict(channel)
        ids = result["input_ids"]
        if ids.shape[-1] > max_seq_length:
            result["input_ids"] = ids[..., :max_seq_length]
            result["attention_mask"] = result["attention_mask"][..., :max_seq_length]
            result["labels"] = result["labels"][..., :max_seq_length]
        result = _fix_duplex_labels(result)

        result["audio_input_lengths"] = _update_audio_lengths_single(
            result["input_ids"],
            result.get("audio_input_lengths"),
            AUDIO_INPUT_PLACEHOLDER.id,
        )
        result["audio_output_lengths"] = _update_audio_lengths_single(
            result["input_ids"],
            result.get("audio_output_lengths"),
            AUDIO_OUTPUT_PLACEHOLDER.id,
        )
        return result

    def filtered_collate_fn(batch: list) -> dict[str, Any]:
        """Collate with sequence length filter and audio truncation."""
        filtered = []
        for channels in batch:
            keep = [_fix_duplex_labels(ch) for ch in channels if ch["input_ids"].shape[-1] <= max_seq_length]
            if keep:
                filtered.append(keep)
        if not filtered:
            # All samples exceed max_seq_length — truncate audio + text together
            non_empty = [chs for chs in batch if chs]
            if non_empty:
                shortest = min(non_empty, key=lambda chs: min(ch["input_ids"].shape[-1] for ch in chs))
                shortest_channel = min(shortest, key=lambda ch: ch["input_ids"].shape[-1])
                filtered.append([_pad_truncate_channel(shortest_channel)])
            else:
                return duplex_collate_fn(batch[:1])  # fallback: pass first sample as-is
        return duplex_collate_fn(filtered)

    def _collect_audio_rows_and_lengths(
        audios: list[torch.Tensor | None],
        lengths: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], torch.Tensor] | tuple[None, None]:
        rows: list[torch.Tensor] = []
        valid_lengths: list[int] = []
        for audio, length in zip(audios, lengths, strict=True):
            if audio is None or length is None:
                continue
            lens = length.to(dtype=torch.long).view(-1)
            if audio.shape[0] != lens.shape[0]:
                raise ValueError(
                    "packed_filtered_collate_fn: audio rows and lengths mismatch "
                    f"({audio.shape[0]} vs {lens.shape[0]})."
                )
            for row, row_len in zip(audio, lens, strict=True):
                row_len_int = int(row_len.item())
                if row_len_int <= 0:
                    continue
                rows.append(row[:row_len_int])
                valid_lengths.append(row_len_int)
        if len(rows) == 0:
            return None, None
        lengths_tensor = torch.tensor(valid_lengths, dtype=torch.long, device=rows[0].device)
        return rows, lengths_tensor

    def _chunk_rows(
        rows: list[torch.Tensor] | None,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if rows is None or lengths is None:
            return None, None
        padded_audio = pad_sequence(rows, batch_first=True, padding_value=0.0)
        return RaonProcessor._chunk_audio(padded_audio, lengths, args.max_audio_seq_length)

    def _sync_speaker_audio(
        input_ids: torch.Tensor,
        speaker_encoder_audio: torch.Tensor | None,
        speaker_encoder_audio_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        num_speaker_tokens = int((input_ids[:, :max_seq_length] == SPEAKER_EMBEDDING_PLACEHOLDER.id).sum().item())
        has_speaker_audio = speaker_encoder_audio is not None and speaker_encoder_audio_lengths is not None
        if not has_speaker_audio:
            return speaker_encoder_audio, speaker_encoder_audio_lengths

        assert speaker_encoder_audio is not None and speaker_encoder_audio_lengths is not None
        assert speaker_encoder_audio.shape[0] == speaker_encoder_audio_lengths.shape[0], (
            "_sync_speaker_audio: speaker audio and lengths must match batch size. "
            f"Got `{speaker_encoder_audio.shape[0]=}` and `{speaker_encoder_audio_lengths.shape[0]=}`."
        )

        speaker_batch_size = int(speaker_encoder_audio.shape[0])
        if speaker_batch_size < num_speaker_tokens:
            pad_count = num_speaker_tokens - speaker_batch_size
            pad_audio = torch.zeros(
                (pad_count, *speaker_encoder_audio.shape[1:]),
                dtype=speaker_encoder_audio.dtype,
                device=speaker_encoder_audio.device,
            )
            pad_lengths = torch.zeros(
                pad_count,
                dtype=speaker_encoder_audio_lengths.dtype,
                device=speaker_encoder_audio_lengths.device,
            )
            speaker_encoder_audio = torch.cat([speaker_encoder_audio, pad_audio], dim=0)
            speaker_encoder_audio_lengths = torch.cat([speaker_encoder_audio_lengths, pad_lengths], dim=0)
        elif speaker_batch_size > num_speaker_tokens:
            speaker_encoder_audio = speaker_encoder_audio[:num_speaker_tokens]
            speaker_encoder_audio_lengths = speaker_encoder_audio_lengths[:num_speaker_tokens]

        return speaker_encoder_audio, speaker_encoder_audio_lengths

    def _collect_and_pad_audio(
        audios: list[torch.Tensor | None],
        lengths: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rows, lengths_tensor = _collect_audio_rows_and_lengths(audios, lengths)
        if rows is None or lengths_tensor is None:
            return None, None
        padded_audio = pad_sequence(rows, batch_first=True, padding_value=0.0)
        return padded_audio, lengths_tensor

    warned_dropped_segments = False

    def packed_filtered_collate_fn(batch: list) -> dict[str, Any]:
        """Filter overlength channels, then pack valid tokens/audio into one sample."""
        nonlocal warned_dropped_segments

        filtered: list[list[dict[str, Any]]] = []
        for channels in batch:
            keep = [_fix_duplex_labels(ch) for ch in channels if ch["input_ids"].shape[-1] <= max_seq_length]
            if keep:
                filtered.append(keep)
        if not filtered:
            non_empty = [chs for chs in batch if chs]
            if non_empty:
                shortest = min(non_empty, key=lambda chs: min(ch["input_ids"].shape[-1] for ch in chs))
                shortest_channel = min(shortest, key=lambda ch: ch["input_ids"].shape[-1])
                filtered = [[_pad_truncate_channel(shortest_channel)]]
            else:
                return duplex_collate_fn(batch[:1])

        flat: list[dict[str, Any]] = [item for channels in filtered for item in channels]
        if len(flat) == 0:
            return duplex_collate_fn(filtered)

        all_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_pos: list[torch.Tensor] = []
        audio_inputs: list[torch.Tensor | None] = []
        audio_input_lengths: list[torch.Tensor | None] = []
        audio_outputs: list[torch.Tensor | None] = []
        audio_output_lengths: list[torch.Tensor | None] = []
        speaker_audios: list[torch.Tensor | None] = []
        speaker_audio_lengths: list[torch.Tensor | None] = []
        segments_per_item: list[list[tuple[int, int]] | None] = []

        for item in flat:
            ids = item["input_ids"]
            labels = item["labels"]
            mask = item["attention_mask"]
            valid = mask[0] == 1
            valid_ids = ids[0][valid]
            valid_labels = labels[0][valid]
            all_ids.append(valid_ids)
            all_labels.append(valid_labels)
            all_pos.append(torch.arange(valid_ids.shape[0], dtype=torch.long))

            audio_inputs.append(item.get("audio_input"))
            audio_input_lengths.append(item.get("audio_input_lengths"))
            audio_outputs.append(item.get("audio_output"))
            audio_output_lengths.append(item.get("audio_output_lengths"))
            speaker_audios.append(item.get("speaker_encoder_audio"))
            speaker_audio_lengths.append(item.get("speaker_encoder_audio_lengths"))
            segments_per_item.append(item.get("audio_output_segments"))

        packed_ids = torch.cat(all_ids)[: args.max_packed_seq_length]
        packed_labels = torch.cat(all_labels)[: args.max_packed_seq_length]
        packed_pos = torch.cat(all_pos)[: args.max_packed_seq_length]
        packed_ids_batch = packed_ids.unsqueeze(0)

        input_rows, input_lengths = _collect_audio_rows_and_lengths(audio_inputs, audio_input_lengths)
        output_rows, output_lengths = _collect_audio_rows_and_lengths(audio_outputs, audio_output_lengths)

        chunked_audio_input, chunked_audio_input_lengths = _chunk_rows(input_rows, input_lengths)
        chunked_audio_output, chunked_audio_output_lengths = _chunk_rows(output_rows, output_lengths)
        chunked_audio_input_lengths = _update_audio_lengths_single(
            input_ids=packed_ids_batch,
            audio_lengths=chunked_audio_input_lengths,
            audio_token_id=AUDIO_INPUT_PLACEHOLDER.id,
        )
        chunked_audio_output_lengths = _update_audio_lengths_single(
            input_ids=packed_ids_batch,
            audio_lengths=chunked_audio_output_lengths,
            audio_token_id=AUDIO_OUTPUT_PLACEHOLDER.id,
        )
        speaker_audio_cat, speaker_audio_len_cat = _collect_and_pad_audio(speaker_audios, speaker_audio_lengths)
        speaker_audio_cat, speaker_audio_len_cat = _sync_speaker_audio(
            input_ids=packed_ids_batch,
            speaker_encoder_audio=speaker_audio_cat,
            speaker_encoder_audio_lengths=speaker_audio_len_cat,
        )

        result: dict[str, Any] = {
            "input_ids": packed_ids_batch,
            "attention_mask": None,
            "position_ids": packed_pos.unsqueeze(0),
            "labels": packed_labels.unsqueeze(0),
            "audio_input": chunked_audio_input,
            "audio_input_lengths": chunked_audio_input_lengths,
            "audio_output": chunked_audio_output,
            "audio_output_lengths": chunked_audio_output_lengths,
            "speaker_encoder_audio": speaker_audio_cat,
            "speaker_encoder_audio_lengths": speaker_audio_len_cat,
        }

        if any(segments is not None for segments in segments_per_item):
            # Current model API supports one segment list per batch.
            # Keep segments only when one sample is packed; otherwise drop.
            if len(flat) == 1 and segments_per_item[0] is not None:
                result["audio_output_segments"] = segments_per_item[0]
            elif not warned_dropped_segments:
                logger.warning(
                    "packed_filtered_collate_fn: Dropping `audio_output_segments` for multi-sample packed batch "
                    "(current model accepts a single segment list)."
                )
                warned_dropped_segments = True

        if speaker_audio_cat is not None and speaker_audio_len_cat is not None:
            result["use_speaker_embedding"] = True

        return result

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        remove_unused_columns=False,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        logging_strategy="steps",
        logging_steps=1,
        dataloader_pin_memory=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=[s.strip() for s in args.report_to.split(",") if s.strip()] or "none",
        ddp_find_unused_parameters=True,
    )

    collator = packed_filtered_collate_fn if args.use_packing else filtered_collate_fn

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        data_collator=collator,
        callbacks=[StepLoggingCallback(), SaveTokenizerCallback(processor)],
    )

    logger.info("Starting duplex fine-tuning ...")
    trainer.train()

    logger.info("Saving model to %s ...", output_dir)
    trainer.save_model(str(output_dir))
    processor.tokenizer.save_pretrained(str(output_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
