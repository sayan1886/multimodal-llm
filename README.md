# Multimodal LLM Project

Implementing Three Architectures: Tools + Adapter + Unified Model

This project contains three complete multi-modal LLM systems

## Project Structure

```mathematica
tree ./
./
├── LICENSE
├── README.md
├── llm-adapter
│   ├── __init__.py
│   ├── adapter.py
│   ├── dataset.py
│   └── train_adapter.py
├── llm-tools
│   ├── __init__.py
│   ├── agent.py
│   ├── flux.py
│   ├── run_agent.py
│   └── whisper.py
├── llm-unified-model
│   ├── __init__.py
│   ├── dataset.py
│   ├── train_unified_model.py
│   └── unified_model.py
├── multimodal_dataset
│   ├── captions.json
│   ├── images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   ├── img3.jpg
│   │   ├── img4.jpg
│   │   └── img5.jpg
│   └── sample_audio.wav
└── requirements.txt

6 directories, 23 files
```

## Dataset

A tiny dataset is added for quick testing of all modules works offline, lightweight, 5 images + JSON + audio.

## Module 1 — LLM + Tools (Whisper + Flux)

LLM acts as an **agent** that decides when to call external tools

- **Whisper** → Speech-to-text
- **Flux** → Image generation
- **LLM** → fallback text reasoning

### Architecture LLM + Tools

```mathematica
User Input → LLM → (Tool Decision)
              ↳ Whisper (if audio request)
              ↳ Flux (if image generation request)
              ↳ LLM direct answer (else)
→ Final Output to User
```

Code Located In - `./llm-tools/`

## Module 2 — LLM + Adapter (Encoder → Adapter → Decoder)

This module uses a **vision encoder** + **adapter MLP** + **decoder LLM**

- **Image** → CLIP Vision Encoder
- **Visual Embedding** → Adapter MLP
- **Token Embeddings** → GPT-2 LLM Decoder

### Architecture LLM + Adapter

```mathematica
Image → Vision Encoder → Adapter → LLM → Text
```

Code Located In - `./llm-adapter/`

## Module 3 — Unified Multimodal Model

This module uses **prefix token** fusion

- Image encoded to a small number of pseudo-tokens
- Pseudo-tokens concatenated with text embeddings
- Passed into the LLM for multimodal reasoning

### Architecture Unified Multimodal Model

```mathematica
Image → Vision Encoder → Prefix Tokens  
Text → Tokenizer  
Concat → LLM → Output
```

Code Located In - `./llm-unified-model/`

## Installation

`pip install -r requirements.txt`

## Requirements

```python
torch
transformers
datasets
tqdm
Pillow
requests
soundfile
numpy
```

## Running Modules

- Run Module 1 (LLM + Tools) `python llm-tools/run_agent.py`
- Run Module 2 (Adapter Fusion) `python llm-adapter/run_adapter.py`
- Run Module 3 (Unified Model) `python llm-unified-model/run_unified_model.py`

Environment variables (image/backends)

- `IMAGE_BACKEND`: choose which image backend the agent will call. Supported values:
    - `flux` (default) — calls `llm-tools/flux.py` which uses OpenAI Images endpoint (requires `FLUX_API_KEY`).
    - `stable-diffusion` / `sd` — uses a local placeholder image generator in `llm-tools/stable-diffusion.py`.
    - `nano-banana` / `nano` — calls `llm-tools/nano_banana.py` which can POST to a configured endpoint or use a placeholder.
- `FLUX_API_KEY`: Bearer API key for the Flux/OpenAI images endpoint (used by `llm-tools/flux.py`).
- `FLUX_MODEL`: optional; override the default image model used by Flux (defaults to `gpt-image-1`).
- `NANO_BANANA_ENDPOINT`: (optional) URL for a Nano-Banana image-generation endpoint that accepts `{"prompt":...}` and returns a base64 image.
- `NANO_BANANA_API_KEY`: (optional) Bearer token for the Nano-Banana endpoint.

Output

- Generated images (placeholder or downloaded) are saved to the project `output/` directory by default, e.g. `./output/sd_image.png` or `./output/nano_image.png`.

Examples

- Run the demo using the Nano-Banana placeholder backend:
    `IMAGE_BACKEND=nano-banana python llm-tools/run_agent.py`
- Run the demo using the stable-diffusion placeholder backend:
    `IMAGE_BACKEND=stable-diffusion python llm-tools/run_agent.py`
- Default (Flux) backend; ensure `FLUX_API_KEY` is exported first:
    ```bash
    export FLUX_API_KEY="sk-..."
    python llm-tools/run_agent.py
    ```

Dataset downsampling utility

If you have a large image dataset and want a quick subset for experiments, use the included script `scripts/downsample_dataset.py`.

Example: create a 1000-image subset using symlinks (fast):

```bash
python scripts/downsample_dataset.py --src multimodal-dataset --dst multimodal-dataset-subset --num 1000 --symlink
```

The script will copy or symlink selected images into `--dst/images/` and will filter `captions.json` if present.

Flickr8k-specific downsampling

If you're working with the Flickr8k CSV-style captions file, there's a convenience script that samples by unique image id and preserves all caption lines for each selected image:

```bash
python scripts/downsample_flickr8k.py --src multimodal-dataset/flickr8k \
    --dst multimodal-dataset/flickr8k-subset --num 1000 --symlink
```

This will create `multimodal-dataset/flickr8k-subset/images/` (symlinks by default) and `multimodal-dataset/flickr8k-subset/captions.txt` containing only caption lines for the selected images.

Environment and `.env`

You can point the adapter/demo scripts to a different dataset using the `MM_DATASET` environment variable. It accepts either:
- a path to a dataset directory containing `captions.txt` (e.g. `./multimodal-dataset/flickr8k-subset`), or
- a known name such as `flickr8k` or `flickr8k-subset` (the repo-level demo will map these to the appropriate folder).

To make this persistent for your shell session, create a repo-root `.env` file with a single line (do not commit secrets):

```bash
# in the repo root
echo "MM_DATASET=multimodal-dataset/flickr8k-subset" > .env
```

The code includes a tiny `.env` loader (no external dependencies) that will read `MM_DATASET` from that file at runtime.

Security note — DO NOT COMMIT SECRETS

The repository's `.gitignore` already ignores `.env`. Keep this pattern: never commit API keys or secrets into the repo. Instead:
- export keys in your shell (e.g. `export FLUX_API_KEY=...`) or
- store them locally in `.env` and ensure `.env` stays out of version control.

## Examples Folder

Contains

```text
example images
audio sample for whisper
```

## License

MIT License — free for academic submission.
