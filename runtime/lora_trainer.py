"""LoRA training on a shared inference GPU.

The problem: vLLM uses ~85% of the L4's 24GB VRAM for inference. We need
to train LoRA adapters in the remaining ~3.6GB.

The solution:
  1. Lower gpu_memory_utilization to 0.85 (reserves ~3.6GB)
  2. Train LoRA rank-8 adapters with gradient checkpointing, batch size 1,
     gradient accumulation over 4 steps (~1GB peak training memory)
  3. Train during idle windows (between user requests)
  4. Save adapter to disk, vLLM hot-loads it via filesystem resolver
  5. Repeat: every N experiences, retrain on the best replay buffer episodes

VRAM budget (L4 = 24GB):
  - Base model + KV cache: ~20.4GB (85% utilization)
  - LoRA adapter weights (rank 8): ~16MB
  - Optimizer states (AdamW): ~32MB
  - Activations (batch=1, grad checkpoint): ~500MB-1GB
  - Headroom: ~2GB
  Total training: ~1.5GB. Fits.

The key insight: we don't train continuously. We accumulate experiences,
format them as instruction-tuning pairs, train a few steps during idle
time, then hot-swap the adapter. The model literally gets better.
"""

import asyncio
import gc
import json
import os
import shutil
import time
import threading
from pathlib import Path

import structlog

from runtime.learning import ReplayBuffer

log = structlog.get_logger()

# LoRA config for Qwen3-8B on L4
LORA_CONFIG = {
    "r": 8,              # rank
    "lora_alpha": 16,    # scaling factor
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "max_steps": 20,        # short bursts
    "warmup_steps": 2,
    "max_seq_length": 1024,  # short to save memory
    "gradient_checkpointing": True,
    "fp16": True,
    "optim": "adamw_torch",
    "save_steps": 20,
    "logging_steps": 5,
}


class LoRATrainer:
    """Trains LoRA adapters on a shared inference GPU during idle windows.

    Lifecycle:
      1. Accumulate experiences in the replay buffer
      2. When enough experiences and GPU is idle, format training data
      3. Train LoRA adapter (rank 8, ~20 steps, ~30 seconds)
      4. Save to adapter_path
      5. vLLM hot-loads via filesystem resolver
    """

    def __init__(
        self,
        model_name: str,
        model_revision: str,
        adapter_path: str,
        replay_buffer: ReplayBuffer,
        min_experiences: int = 20,
        retrain_interval: int = 50,  # retrain every N new experiences
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.adapter_path = Path(adapter_path)
        self.replay_buffer = replay_buffer
        self.min_experiences = min_experiences
        self.retrain_interval = retrain_interval

        self.adapter_path.mkdir(parents=True, exist_ok=True)

        self.training_count = 0
        self.last_train_at_experience = 0
        self.is_training = False
        self._lock = threading.Lock()

        # Check if we have a pre-existing adapter
        self.has_adapter = (self.adapter_path / "adapter_config.json").exists()

        log.info(
            "lora_trainer_initialized",
            adapter_path=str(self.adapter_path),
            has_existing_adapter=self.has_adapter,
        )

    def should_train(self) -> bool:
        """Check if we should kick off a training run."""
        if self.is_training:
            return False
        if self.replay_buffer.size < self.min_experiences:
            return False
        experiences_since_last = self.replay_buffer.total_added - self.last_train_at_experience
        return experiences_since_last >= self.retrain_interval

    def format_training_data(self, max_samples: int = 100) -> list[dict]:
        """Format replay buffer experiences as instruction-tuning pairs.

        Selects the highest-reward experiences and formats them as
        chat-style training examples: user message -> ideal response.
        """
        if self.replay_buffer.size == 0:
            return []

        # Sample more than we need, sort by reward
        sample_size = min(self.replay_buffer.size, max_samples * 3)
        batch, _ = self.replay_buffer.sample(sample_size)
        batch.sort(key=lambda e: e.reward, reverse=True)

        # Take the best
        good_episodes = [e for e in batch if e.reward > 0.3][:max_samples]

        training_data = []
        for ep in good_episodes:
            # Parse the state_summary for the user input
            user_text = ep.state_summary
            if ": " in user_text:
                user_text = user_text.split(": ", 1)[1]

            training_data.append({
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": ep.observation},
                ],
            })

        log.info("training_data_formatted", samples=len(training_data),
                 mean_reward=sum(e.reward for e in good_episodes) / max(1, len(good_episodes)))
        return training_data

    def train_sync(self) -> dict:
        """Run a LoRA training step. Blocks until complete.

        This is the core function that actually touches the GPU.
        Designed to run in ~30-60 seconds on an L4.
        """
        with self._lock:
            if self.is_training:
                return {"skipped": True, "reason": "already training"}
            self.is_training = True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model, TaskType

            training_data = self.format_training_data()
            if len(training_data) < 5:
                return {"skipped": True, "reason": f"only {len(training_data)} samples"}

            start = time.time()
            log.info("lora_training_start", samples=len(training_data))

            # Write training data to a temp jsonl file
            data_path = self.adapter_path / "train_data.jsonl"
            with open(data_path, "w") as f:
                for item in training_data:
                    f.write(json.dumps(item) + "\n")

            # Load tokenizer (CPU, no VRAM)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision, trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with minimal GPU memory
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # Key: only load for training, not for full inference
                low_cpu_mem_usage=True,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                r=LORA_CONFIG["r"],
                lora_alpha=LORA_CONFIG["lora_alpha"],
                lora_dropout=LORA_CONFIG["lora_dropout"],
                target_modules=LORA_CONFIG["target_modules"],
                bias=LORA_CONFIG["bias"],
                task_type=TaskType.CAUSAL_LM,
            )

            # If we have a previous adapter, load it and continue training
            if self.has_adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(self.adapter_path))
                model.train()
                log.info("continuing_from_existing_adapter")
            else:
                model = get_peft_model(model, lora_config)
                model.train()

            model.print_trainable_parameters()

            # Tokenize training data
            def tokenize_messages(messages: list[dict]) -> dict:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer(
                    text, truncation=True,
                    max_length=TRAINING_CONFIG["max_seq_length"],
                    padding="max_length", return_tensors="pt",
                )
                tokens["labels"] = tokens["input_ids"].clone()
                return {k: v.squeeze(0) for k, v in tokens.items()}

            from torch.utils.data import Dataset

            class ChatDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                    self.tokenized = [tokenize_messages(d["messages"]) for d in data]

                def __len__(self):
                    return len(self.tokenized)

                def __getitem__(self, idx):
                    return self.tokenized[idx]

            dataset = ChatDataset(training_data)

            # Training arguments -- minimal footprint
            output_dir = str(self.adapter_path / "checkpoints")
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
                num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
                max_steps=TRAINING_CONFIG["max_steps"],
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                fp16=TRAINING_CONFIG["fp16"],
                gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
                optim=TRAINING_CONFIG["optim"],
                save_steps=TRAINING_CONFIG["save_steps"],
                logging_steps=TRAINING_CONFIG["logging_steps"],
                save_total_limit=1,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,  # save memory
            )

            from transformers import Trainer

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )

            # Train
            train_result = trainer.train()

            # Save the adapter
            model.save_pretrained(str(self.adapter_path))
            tokenizer.save_pretrained(str(self.adapter_path))

            # Write adapter_model_config for vLLM compatibility
            adapter_config = {
                "base_model_name_or_path": self.model_name,
                "revision": self.model_revision,
            }
            (self.adapter_path / "adapter_model_config.json").write_text(
                json.dumps(adapter_config)
            )

            # Clean up GPU memory
            del model, trainer, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - start
            self.training_count += 1
            self.last_train_at_experience = self.replay_buffer.total_added
            self.has_adapter = True

            metrics = {
                "training_count": self.training_count,
                "samples": len(training_data),
                "loss": train_result.training_loss,
                "elapsed_seconds": round(elapsed, 1),
                "adapter_path": str(self.adapter_path),
            }
            log.info("lora_training_complete", **metrics)

            # Clean up checkpoints dir
            shutil.rmtree(output_dir, ignore_errors=True)
            data_path.unlink(missing_ok=True)

            return metrics

        except ImportError as e:
            log.warning("lora_training_unavailable", missing=str(e))
            return {"skipped": True, "reason": f"Missing dependency: {e}"}

        except Exception as e:
            log.error("lora_training_failed", error=str(e))
            return {"error": str(e)}

        finally:
            self.is_training = False
            # Always try to free GPU memory
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def train_async(self) -> dict:
        """Run training in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.train_sync)

    def summary(self) -> dict:
        return {
            "training_count": self.training_count,
            "has_adapter": self.has_adapter,
            "is_training": self.is_training,
            "last_train_at_experience": self.last_train_at_experience,
            "next_train_at": self.last_train_at_experience + self.retrain_interval,
            "replay_buffer_size": self.replay_buffer.size,
            "adapter_path": str(self.adapter_path),
        }
