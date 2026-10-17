"""Tests for the LoRA trainer module."""

from runtime.learning import ReplayBuffer, Experience
from runtime.lora_trainer import LoRATrainer, LORA_CONFIG, TRAINING_CONFIG


class TestLoRATrainer:
    def _make_buffer(self, n: int = 30) -> ReplayBuffer:
        rb = ReplayBuffer(capacity=100)
        for i in range(n):
            rb.add(Experience(
                f"step {i}: question {i}", "communicate",
                {"input": f"q{i}"}, f"answer {i}",
                reward=0.5 + (i % 3) * 0.15, novelty=0.5,
            ))
        return rb

    def test_should_train_not_enough(self):
        rb = ReplayBuffer(capacity=100)
        for i in range(5):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}", 0.5))
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb,
                              min_experiences=20, retrain_interval=50)
        assert not trainer.should_train()

    def test_should_train_ready(self):
        rb = self._make_buffer(60)
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb,
                              min_experiences=20, retrain_interval=50)
        assert trainer.should_train()

    def test_should_train_already_training(self):
        rb = self._make_buffer(60)
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb,
                              min_experiences=20, retrain_interval=50)
        trainer.is_training = True
        assert not trainer.should_train()

    def test_format_training_data(self):
        rb = self._make_buffer(30)
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb)
        data = trainer.format_training_data(max_samples=10)
        assert len(data) > 0
        assert len(data) <= 10
        # Each sample should have messages
        for sample in data:
            assert "messages" in sample
            assert len(sample["messages"]) == 2
            assert sample["messages"][0]["role"] == "user"
            assert sample["messages"][1]["role"] == "assistant"

    def test_format_training_data_filters_low_reward(self):
        rb = ReplayBuffer(capacity=100)
        for i in range(20):
            rb.add(Experience(f"s{i}", "t", {}, f"o{i}",
                              reward=0.1))  # all low reward
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb)
        data = trainer.format_training_data()
        # Low reward episodes should be filtered out
        assert len(data) == 0

    def test_summary(self):
        rb = self._make_buffer()
        trainer = LoRATrainer("model", "rev", "/tmp/test-lora", rb)
        s = trainer.summary()
        assert s["training_count"] == 0
        assert s["has_adapter"] is False
        assert s["is_training"] is False


class TestLoRAConfig:
    def test_config_values(self):
        assert LORA_CONFIG["r"] == 8
        assert LORA_CONFIG["lora_alpha"] == 16
        assert "q_proj" in LORA_CONFIG["target_modules"]

    def test_training_config(self):
        assert TRAINING_CONFIG["per_device_train_batch_size"] == 1
        assert TRAINING_CONFIG["gradient_checkpointing"] is True
        assert TRAINING_CONFIG["fp16"] is True
