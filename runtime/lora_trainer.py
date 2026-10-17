"""Opt-in, evaluation-gated LoRA adaptation.

Online weight updates are disabled by default by the runtime. When an operator
explicitly enables them, this trainer curates high-reward examples, keeps a
held-out split, trains into a staging directory, rejects regressions, versions
the previous adapter, and rolls back if vLLM cannot load the candidate.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from runtime.learning import ReplayBuffer

log = structlog.get_logger()

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 20,
    "warmup_steps": 2,
    "max_seq_length": 1024,
    "gradient_checkpointing": True,
    "fp16": True,
}
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]\s*\S+"
)


class LoRATrainer:
    """Train and promote adapters only when a held-out evaluation gate passes."""

    def __init__(
        self,
        model_name: str,
        model_revision: str,
        adapter_path: str,
        replay_buffer: ReplayBuffer,
        min_experiences: int = 20,
        retrain_interval: int = 50,
        minimum_reward: float = 0.6,
        validation_fraction: float = 0.2,
        max_validation_loss_regression: float = 0.02,
        retain_versions: int = 3,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.adapter_path = Path(adapter_path)
        self.version_path = self.adapter_path.parent / f"{self.adapter_path.name}-versions"
        self.audit_path = self.adapter_path.parent / f"{self.adapter_path.name}-training-audit.jsonl"
        self.replay_buffer = replay_buffer
        self.min_experiences = max(5, min_experiences)
        self.retrain_interval = max(1, retrain_interval)
        self.minimum_reward = max(-1.0, min(float(minimum_reward), 1.0))
        self.validation_fraction = max(0.1, min(float(validation_fraction), 0.5))
        self.max_validation_loss_regression = max(
            0.0, min(float(max_validation_loss_regression), 0.25)
        )
        self.retain_versions = max(1, min(int(retain_versions), 10))
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.version_path.mkdir(parents=True, exist_ok=True)
        self.training_count = 0
        self.last_train_at_experience = 0
        self.is_training = False
        self._lock = threading.Lock()
        self.has_adapter = (self.adapter_path / "adapter_config.json").exists()

    def should_train(self) -> bool:
        if self.is_training or self.replay_buffer.size < self.min_experiences:
            return False
        return (
            self.replay_buffer.total_added - self.last_train_at_experience
            >= self.retrain_interval
        )

    def format_training_data(self, max_samples: int = 100) -> list[dict[str, Any]]:
        """Deterministically curate high-reward, deduplicated, secret-free examples."""
        if self.replay_buffer.size == 0:
            return []
        episodes = list(self.replay_buffer.buffer[: self.replay_buffer.size])
        episodes.sort(key=lambda episode: (-episode.reward, episode.id))
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for episode in episodes:
            if episode.reward < self.minimum_reward:
                continue
            user_text = episode.state_summary.split(": ", 1)[-1].strip()[:4_000]
            assistant_text = episode.observation.strip()[:8_000]
            if not user_text or not assistant_text:
                continue
            combined = f"{user_text}\n{assistant_text}"
            if SECRET_PATTERN.search(combined):
                continue
            digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            output.append(
                {
                    "id": episode.id,
                    "reward": episode.reward,
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                }
            )
            if len(output) >= max_samples:
                break
        return output

    def split_training_data(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Create a stable holdout split so repeated runs are comparable."""
        train: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        threshold = int(self.validation_fraction * 10_000)
        for item in data:
            bucket = int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16) % 10_000
            (validation if bucket < threshold else train).append(item)
        if data and not validation:
            validation.append(train.pop())
        if len(data) > 1 and not train:
            train.append(validation.pop())
        return train, validation

    def _audit(self, event: dict[str, Any]) -> None:
        event = {"timestamp": time.time(), **event}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _gpu_has_headroom(self, torch_module) -> tuple[bool, float]:
        if not torch_module.cuda.is_available():
            return False, 0.0
        free_bytes, _ = torch_module.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        # An 8B NF4 model plus activations and allocator overhead needs more than
        # the previously claimed 3.6 GB. Refuse instead of risking inference OOM.
        return free_gb >= 6.0, round(free_gb, 2)

    def train_sync(self) -> dict[str, Any]:
        with self._lock:
            if self.is_training:
                return {"skipped": True, "reason": "already training"}
            self.is_training = True

        staging = self.adapter_path.parent / f".{self.adapter_path.name}-candidate-{uuid.uuid4().hex[:8]}"
        try:
            import torch
            from peft import (
                LoraConfig,
                PeftModel,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from torch.utils.data import Dataset
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                Trainer,
                TrainingArguments,
            )

            enough_memory, free_gb = self._gpu_has_headroom(torch)
            if not enough_memory:
                result = {"skipped": True, "reason": "insufficient GPU headroom", "free_gb": free_gb}
                self._audit({"event": "skipped", **result})
                return result

            data = self.format_training_data()
            train_data, validation_data = self.split_training_data(data)
            if len(train_data) < 5 or len(validation_data) < 1:
                result = {
                    "skipped": True,
                    "reason": "insufficient curated train/validation samples",
                    "train_samples": len(train_data),
                    "validation_samples": len(validation_data),
                }
                self._audit({"event": "skipped", **result})
                return result

            started = time.time()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                quantization_config=quantization,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"]
            )
            if self.has_adapter:
                model = PeftModel.from_pretrained(model, str(self.adapter_path), is_trainable=True)
            else:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=LORA_CONFIG["r"],
                        lora_alpha=LORA_CONFIG["lora_alpha"],
                        lora_dropout=LORA_CONFIG["lora_dropout"],
                        target_modules=LORA_CONFIG["target_modules"],
                        bias=LORA_CONFIG["bias"],
                        task_type=TaskType.CAUSAL_LM,
                    ),
                )

            def tokenize(item: dict[str, Any]) -> dict[str, Any]:
                messages = item["messages"]
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt_text = tokenizer.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
                tokens = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=TRAINING_CONFIG["max_seq_length"],
                    padding="max_length",
                    return_tensors="pt",
                )
                prompt_length = len(
                    tokenizer(
                        prompt_text,
                        truncation=True,
                        max_length=TRAINING_CONFIG["max_seq_length"],
                    )["input_ids"]
                )
                labels = tokens["input_ids"].clone()
                labels[:, :prompt_length] = -100
                labels[tokens["attention_mask"] == 0] = -100
                tokens["labels"] = labels
                return {key: value.squeeze(0) for key, value in tokens.items()}

            class ChatDataset(Dataset):
                def __init__(self, rows: list[dict[str, Any]]):
                    self.rows = [tokenize(row) for row in rows]

                def __len__(self) -> int:
                    return len(self.rows)

                def __getitem__(self, index: int) -> dict[str, Any]:
                    return self.rows[index]

            train_dataset = ChatDataset(train_data)
            validation_dataset = ChatDataset(validation_data)
            training_args = TrainingArguments(
                output_dir=str(staging / "checkpoints"),
                per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
                max_steps=TRAINING_CONFIG["max_steps"],
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                fp16=TRAINING_CONFIG["fp16"],
                gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
                optim="paged_adamw_8bit",
                logging_steps=5,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
            )
            baseline_loss = float(trainer.evaluate()["eval_loss"])
            train_result = trainer.train()
            candidate_loss = float(trainer.evaluate()["eval_loss"])
            loss_limit = baseline_loss * (1.0 + self.max_validation_loss_regression)
            gate_passed = math.isfinite(candidate_loss) and candidate_loss <= loss_limit
            metrics = {
                "train_samples": len(train_data),
                "validation_samples": len(validation_data),
                "training_loss": float(train_result.training_loss),
                "baseline_validation_loss": baseline_loss,
                "candidate_validation_loss": candidate_loss,
                "validation_loss_limit": loss_limit,
                "gate_passed": gate_passed,
                "elapsed_seconds": round(time.time() - started, 1),
            }
            if not gate_passed:
                self._audit({"event": "rejected", **metrics})
                return {"rejected": True, **metrics}

            staging.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(staging)
            tokenizer.save_pretrained(staging)
            shutil.rmtree(staging / "checkpoints", ignore_errors=True)
            (staging / "evaluation.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            promoted, reason = self._promote_candidate(staging)
            if not promoted:
                self._audit({"event": "rollback", "reason": reason, **metrics})
                return {"error": reason, "rolled_back": True, **metrics}

            self.training_count += 1
            self.last_train_at_experience = self.replay_buffer.total_added
            self.has_adapter = True
            result = {"training_count": self.training_count, "promoted": True, **metrics}
            self._audit({"event": "promoted", **result})
            return result
        except ImportError as exc:
            result = {"skipped": True, "reason": f"Missing training dependency: {exc}"}
            self._audit({"event": "skipped", **result})
            return result
        except Exception as exc:
            log.exception("lora_training_failed")
            result = {"error": str(exc)}
            self._audit({"event": "failed", **result})
            return result
        finally:
            self.is_training = False
            shutil.rmtree(staging, ignore_errors=True)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _promote_candidate(self, staging: Path) -> tuple[bool, str]:
        version: Path | None = None
        if self.adapter_path.exists():
            version = self.version_path / time.strftime("%Y%m%d-%H%M%S")
            suffix = 1
            while version.exists():
                version = self.version_path / f"{version.name}-{suffix}"
                suffix += 1
            shutil.copytree(self.adapter_path, version)
            shutil.rmtree(self.adapter_path)
        os.replace(staging, self.adapter_path)
        if not self._hot_load_adapter():
            shutil.rmtree(self.adapter_path, ignore_errors=True)
            if version:
                shutil.copytree(version, self.adapter_path)
                self._hot_load_adapter()
            return False, "vLLM rejected candidate adapter"
        versions = sorted(
            (path for path in self.version_path.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_version in versions[self.retain_versions :]:
            shutil.rmtree(old_version, ignore_errors=True)
        return True, "ok"

    def _hot_load_adapter(self) -> bool:
        import httpx

        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
        adapter_name = "agent-lora"
        try:
            try:
                httpx.post(
                    f"{base_url}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
                    timeout=10,
                )
            except httpx.HTTPError:
                pass
            response = httpx.post(
                f"{base_url}/v1/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": str(self.adapter_path)},
                timeout=30,
            )
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    async def train_async(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.train_sync)

    def summary(self) -> dict[str, Any]:
        return {
            "training_count": self.training_count,
            "has_adapter": self.has_adapter,
            "is_training": self.is_training,
            "last_train_at_experience": self.last_train_at_experience,
            "next_train_at": self.last_train_at_experience + self.retrain_interval,
            "replay_buffer_size": self.replay_buffer.size,
            "minimum_reward": self.minimum_reward,
            "validation_fraction": self.validation_fraction,
            "adapter_path": str(self.adapter_path),
        }
