# Weights Directory

This directory holds model weights or references to externally hosted weights.

In the skeleton agent, no weights are included. A real `.agi` file would contain
or reference:

- **base_llm.safetensors** -- Main language model (~4-400GB depending on quantization)
- **vision_encoder.safetensors** -- Vision transformer for image perception (~1-5GB)
- **audio_encoder.safetensors** -- Audio encoder for speech/sound (~1-3GB)
- **world_model.safetensors** -- Learned dynamics model (~0.5-2GB)
- **tactic_predictor.safetensors** -- Lean4 tactic predictor for formal proofs (~0.5GB)
- **tokenizer.json** -- Tokenizer vocabulary and merge rules (~5MB)

## External Weight References

For portability, weights can be referenced externally via `weights.json`:

```json
{
  "models": [
    {
      "name": "base_llm",
      "format": "safetensors",
      "quantization": "q4_k_m",
      "sha256": "...",
      "size_bytes": 4200000000,
      "remote": "https://huggingface.co/example/model/resolve/main/model.safetensors"
    }
  ]
}
```

The runtime downloads and caches weights on first load, verifying integrity via SHA-256.
