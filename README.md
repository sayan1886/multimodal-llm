# Multimodal LLM — Project README

This repository contains reference code for three multimodal approaches and utility scripts to run demos and manage datasets.

Repository layout (high level)
- `llm-tools/` — agent, Whisper ASR, image backends (Flux, Stable Diffusion adapters, Nano-Banana, DeepInfra, Freepik).
- `llm-adapter/` — CLIP encoder + Adapter MLP + GPT-2 decoder implementation and training utilities.
- `llm-unified-model/` — unified-prefix multimodal model training and demo runner.
- `multimodal-dataset/` — small example dataset included for quick tests.
- `scripts/` — dataset helper scripts (downsampling for generic sets and Flickr8k CSV).

Quick setup
1. Create a Python virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Running the demos
- LLM + Tools agent (ASR + image backend + LLM):
    ```bash
    python llm-tools/run_agent.py
    ```
    - The agent uses `Whisper` for ASR and an image backend selected via `IMAGE_BACKEND`.

- Adapter training (vision → adapter → LLM):
    ```bash
    python llm-adapter/train_adapter.py
    ```

- Unified model training + demo:
    ```bash
    python llm-unified-model/run_unified_model.py
    ```

Environment variables
- `MM_DATASET`: dataset path or known name (`flickr8k`, `flickr8k-subset`). If set to a directory containing `captions.txt`, the runners will use that folder.
- `IMAGE_BACKEND`: which image backend to use (default `flux`). Supported backends: `flux`, `stable-diffusion`/`sd`, `nano-banana`, `deepinfra`, `freepik`.

Common backend env vars (examples)
- `FLUX_API_KEY`, `FLUX_MODEL`
- `STABILITY_API_KEY`, `HF_API_KEY`, `A1111_URL`, `REPLICATE_API_TOKEN`
- `NANO_BANANA_API_KEY`, `NANO_BANANA_MODEL`
- `DEEP_INFRA_API_KEY`, `FREEPIK_API_KEY`

Dataset helpers
- Generic downsample (JSON/images):
    ```bash
    python scripts/downsample_dataset.py --src <src> --dst <dst> --num <N> --symlink
    ```
- Flickr8k CSV downsample (preserves all captions for sampled images):
    ```bash
    python scripts/downsample_flickr8k.py --src multimodal-dataset/flickr8k \
            --dst multimodal-dataset/flickr8k-subset --num 1000 --symlink
    ```

Outputs and checkpoints
- Generated images are written to `./output/` by the image backend modules.
- Training checkpoints are written to module-specific `model_checkpoints/` folders.

Security / best practices
- Never commit API keys or secrets. Use a repo-root `.env` (ignored by `.gitignore`) or export keys in your shell.
- Use `.env.example` as a template and copy it to `.env` with real values locally.

What's next (optional)
- I can remove any real API keys from the repo `.env` and replace them with placeholders, create more targeted run scripts for experiments, or run a quick smoke test of a demo (note: training runs are long).

---
