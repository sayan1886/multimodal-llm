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
User → LLM → Tool Decision  
      → Whisper / Flux → Result → LLM → Final Output
```

Code Located In - `./llm_tools/`

## Module 2 — LLM + Adapter (Encoder → Adapter → Decoder)

This module uses a **vision encoder** + **adapter MLP** + **decoder LLM**

- **Image** → CLIP Vision Encoder
- **Visual Embedding** → Adapter MLP
- **Token Embeddings** → GPT-2 LLM Decoder

### Architecture LLM + Adapter

```mathematica
Image → Vision Encoder → Adapter → LLM → Text
```

Code Located In - `./llm_adapter/`

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

Code Located In - `./module3_llm-unified-model/`

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

- Run Module 1 (LLM + Tools) `python llm_tools/demo_module1.py`
- Run Module 2 (Adapter Fusion) `python llm_adapter/train_adapter.py`
- Run Module 3 (Unified Model) `python llm_unified_model/train_unified.py`

## Examples Folder

Contains

```text
example images
audio sample for whisper
```

## License

MIT License — free for academic submission.
