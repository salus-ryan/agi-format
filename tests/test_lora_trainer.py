"""Tests for evaluation-gated LoRA training."""

from runtime.learning import Experience, ReplayBuffer
from runtime.lora_trainer import LORA_CONFIG, TRAINING_CONFIG, LoRATrainer


class TestLoRATrainer:
    def _make_buffer(self, n: int = 30) -> ReplayBuffer:
        buffer = ReplayBuffer(capacity=100)
        for index in range(n):
            buffer.add(
                Experience(
                    f"step {index}: question {index}",
                    "communicate",
                    {"input": f"q{index}"},
                    f"answer {index}",
                    reward=0.5 + (index % 3) * 0.15,
                    novelty=0.5,
                )
            )
        return buffer

    @staticmethod
    def _trainer(tmp_path, buffer, **kwargs):
        return LoRATrainer("model", "rev", str(tmp_path / "adapter"), buffer, **kwargs)

    def test_should_train_not_enough(self, tmp_path):
        buffer = ReplayBuffer(capacity=100)
        for index in range(5):
            buffer.add(Experience(f"s{index}", "t", {}, f"o{index}", 0.5))
        trainer = self._trainer(
            tmp_path, buffer, min_experiences=20, retrain_interval=50
        )
        assert not trainer.should_train()

    def test_should_train_ready(self, tmp_path):
        trainer = self._trainer(
            tmp_path,
            self._make_buffer(60),
            min_experiences=20,
            retrain_interval=50,
        )
        assert trainer.should_train()

    def test_should_train_already_training(self, tmp_path):
        trainer = self._trainer(
            tmp_path,
            self._make_buffer(60),
            min_experiences=20,
            retrain_interval=50,
        )
        trainer.is_training = True
        assert not trainer.should_train()

    def test_format_training_data(self, tmp_path):
        trainer = self._trainer(tmp_path, self._make_buffer(30))
        data = trainer.format_training_data(max_samples=10)
        assert 0 < len(data) <= 10
        for sample in data:
            assert len(sample["messages"]) == 2
            assert sample["messages"][0]["role"] == "user"
            assert sample["messages"][1]["role"] == "assistant"

    def test_format_training_data_filters_low_reward(self, tmp_path):
        buffer = ReplayBuffer(capacity=100)
        for index in range(20):
            buffer.add(Experience(f"s{index}", "t", {}, f"o{index}", reward=0.1))
        trainer = self._trainer(tmp_path, buffer)
        assert trainer.format_training_data() == []

    def test_format_training_data_filters_secrets(self, tmp_path):
        buffer = ReplayBuffer(capacity=10)
        buffer.add(
            Experience(
                "step 1: API_KEY=top-secret", "communicate", {}, "done", reward=1.0
            )
        )
        trainer = self._trainer(tmp_path, buffer)
        assert trainer.format_training_data() == []

    def test_split_is_stable_and_has_holdout(self, tmp_path):
        trainer = self._trainer(tmp_path, self._make_buffer(30))
        data = trainer.format_training_data()
        first = trainer.split_training_data(data)
        second = trainer.split_training_data(data)
        assert [item["id"] for item in first[0]] == [item["id"] for item in second[0]]
        assert first[0]
        assert first[1]

    def test_summary(self, tmp_path):
        trainer = self._trainer(tmp_path, self._make_buffer())
        summary = trainer.summary()
        assert summary["training_count"] == 0
        assert summary["has_adapter"] is False
        assert summary["is_training"] is False
        assert summary["validation_fraction"] == 0.2


class TestLoRAConfig:
    def test_config_values(self):
        assert LORA_CONFIG["r"] == 8
        assert LORA_CONFIG["lora_alpha"] == 16
        assert "q_proj" in LORA_CONFIG["target_modules"]

    def test_training_config(self):
        assert TRAINING_CONFIG["per_device_train_batch_size"] == 1
        assert TRAINING_CONFIG["gradient_checkpointing"] is True
        assert TRAINING_CONFIG["fp16"] is True
