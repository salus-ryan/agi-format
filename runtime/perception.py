"""Perception adapters -- convert raw multimodal input into Percepts.

Adapters:
  - TextAdapter: tokenization + embedding (always available)
  - VisionAdapter: image encoding via SigLIP (requires GPU + model)
  - AudioAdapter: speech/sound encoding via Whisper (requires GPU + model)
  - StructuredAdapter: JSON/CSV/tabular data (always available)

Each adapter is optional. The PerceptionPipeline auto-detects input
modality and routes to the right adapter.
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

from runtime.types import Percept

log = structlog.get_logger()


class TextAdapter:
    """Convert text input into a Percept with pseudo-embedding."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def process(self, text: str) -> Percept:
        embedding = self._embed(text)
        return Percept(
            modality="text",
            content=text,
            embedding=embedding.tolist(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "length": len(text),
                "word_count": len(text.split()),
                "language": self._detect_language(text),
            },
        )

    def _embed(self, text: str) -> np.ndarray:
        """Character trigram hash embedding. Fast, deterministic, good enough
        for similarity. A real implementation would use sentence-transformers."""
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            h = hash(trigram)
            vec[h % self.embedding_dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _detect_language(text: str) -> str:
        """Heuristic language detection. Real: use fasttext lid."""
        # Check for common non-ASCII scripts
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        arabic = sum(1 for c in text if '\u0600' <= c <= '\u06ff')

        total = len(text) or 1
        if cjk / total > 0.1:
            return "zh"
        if cyrillic / total > 0.1:
            return "ru"
        if arabic / total > 0.1:
            return "ar"
        return "en"


class VisionAdapter:
    """Convert image input into a Percept.

    Supports two modes:
    1. With a real vision encoder (SigLIP via transformers) -- produces
       dense embeddings for semantic image understanding
    2. Without (fallback) -- produces a content hash and basic metadata

    The real encoder is loaded lazily on first use.
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def process(self, image_data: bytes, filename: str = "image") -> Percept:
        content_hash = hashlib.sha256(image_data).hexdigest()[:16]
        metadata = {
            "filename": filename,
            "size_bytes": len(image_data),
            "content_hash": content_hash,
        }

        # Try to get image dimensions
        try:
            metadata.update(self._get_image_info(image_data))
        except Exception:
            pass

        # Try real vision encoding
        embedding = None
        try:
            embedding = self._encode(image_data)
            metadata["encoder"] = self.model_name
        except Exception as e:
            log.debug("vision_encoder_unavailable", error=str(e))
            metadata["encoder"] = "hash_only"

        return Percept(
            modality="vision",
            content=f"image:{content_hash}",
            embedding=embedding,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def _encode(self, image_data: bytes) -> list[float] | None:
        """Encode image with SigLIP. Lazy-loads the model."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                log.info("vision_encoder_loaded", model=self.model_name)
            except ImportError:
                log.warning("transformers_not_available")
                return None

        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_data))
        inputs = self._processor(images=img, return_tensors="pt")
        outputs = self._model.get_image_features(**inputs)
        embedding = outputs[0].detach().numpy().tolist()
        return embedding

    @staticmethod
    def _get_image_info(data: bytes) -> dict:
        """Extract basic image info from headers without PIL."""
        info = {}
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            info["format"] = "png"
            if len(data) > 24:
                import struct
                w, h = struct.unpack(">II", data[16:24])
                info["width"], info["height"] = w, h
        # JPEG
        elif data[:2] == b'\xff\xd8':
            info["format"] = "jpeg"
        # WebP
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            info["format"] = "webp"
        return info


class AudioAdapter:
    """Convert audio input into a Percept.

    With Whisper: transcribes audio and produces an embedding.
    Without: produces content hash and basic metadata.
    """

    def __init__(self, model_name: str = "openai/whisper-base"):
        self.model_name = model_name
        self._pipeline = None

    def process(self, audio_data: bytes, filename: str = "audio", sample_rate: int = 16000) -> Percept:
        content_hash = hashlib.sha256(audio_data).hexdigest()[:16]
        metadata = {
            "filename": filename,
            "size_bytes": len(audio_data),
            "content_hash": content_hash,
            "sample_rate": sample_rate,
        }

        transcription = None
        try:
            transcription = self._transcribe(audio_data, sample_rate)
            metadata["transcription"] = transcription
            metadata["encoder"] = self.model_name
        except Exception as e:
            log.debug("audio_encoder_unavailable", error=str(e))
            metadata["encoder"] = "hash_only"

        return Percept(
            modality="audio",
            content=transcription or f"audio:{content_hash}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def _transcribe(self, audio_data: bytes, sample_rate: int) -> str | None:
        """Transcribe audio with Whisper. Lazy-loads the pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                )
                log.info("audio_encoder_loaded", model=self.model_name)
            except ImportError:
                return None

        import io
        import numpy as np

        # Convert raw bytes to float32 array (assuming 16-bit PCM)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._pipeline({"raw": audio_array, "sampling_rate": sample_rate})
        return result.get("text", "")


class StructuredAdapter:
    """Convert structured data (dict, JSON, CSV rows) into a Percept."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def process(self, data: dict | list) -> Percept:
        text_repr = json.dumps(data, default=str)[:5000]
        embedding = self._embed(text_repr)

        if isinstance(data, dict):
            metadata = {
                "keys": list(data.keys())[:50],
                "key_count": len(data),
                "nested": any(isinstance(v, (dict, list)) for v in data.values()),
            }
        elif isinstance(data, list):
            metadata = {
                "length": len(data),
                "item_type": type(data[0]).__name__ if data else "empty",
            }
        else:
            metadata = {"type": type(data).__name__}

        return Percept(
            modality="structured",
            content=data,
            embedding=embedding.tolist(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for i in range(len(text) - 2):
            trigram = text[i:i+3].lower()
            h = hash(trigram)
            vec[h % self.embedding_dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class PerceptionPipeline:
    """Routes input to the right adapter based on type detection."""

    def __init__(self, embedding_dim: int = 128):
        self.text = TextAdapter(embedding_dim)
        self.vision = VisionAdapter()
        self.audio = AudioAdapter()
        self.structured = StructuredAdapter(embedding_dim)
        self.percept_count = 0

    def perceive(self, raw_input: str | bytes | dict | list | None) -> list[Percept]:
        """Auto-detect modality and produce Percepts."""
        if raw_input is None:
            return []

        self.percept_count += 1

        if isinstance(raw_input, str):
            return [self.text.process(raw_input)]

        if isinstance(raw_input, bytes):
            # Detect image vs audio from magic bytes
            if raw_input[:8] == b'\x89PNG\r\n\x1a\n' or raw_input[:2] == b'\xff\xd8':
                return [self.vision.process(raw_input)]
            if raw_input[:4] == b'RIFF':
                fmt = raw_input[8:12]
                if fmt == b'WEBP':
                    return [self.vision.process(raw_input)]
                elif fmt == b'WAVE':
                    return [self.audio.process(raw_input)]
            # Default: treat as audio PCM
            return [self.audio.process(raw_input)]

        if isinstance(raw_input, (dict, list)):
            return [self.structured.process(raw_input)]

        log.warning("unknown_input_type", type=type(raw_input).__name__)
        return []
