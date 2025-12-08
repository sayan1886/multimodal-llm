# llm-tools/stable_diffusion.py

import os
import base64
import requests
from PIL import Image
from io import BytesIO


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_image_bytes(img_bytes, out_path):
    """Save decoded bytes to disk."""
    if not os.path.isabs(out_path):
        out_path = os.path.join(OUTPUT_DIR, out_path)

    img = Image.open(BytesIO(img_bytes))
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------
# 1. Stability AI — Stable Diffusion 3 or SDXL
# ---------------------------------------------------------------------

def _generate_stability(prompt, out_path):
    api_key = os.getenv("STABILITY_API_KEY")
    model = os.getenv("STABILITY_MODEL", "sd3")  # sd3 or sd3-turbo

    if not api_key:
        raise ValueError("STABILITY_API_KEY missing.")

    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg_scale": 7,
        "samples": 1,
        "output_format": "png"
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()

    data = resp.json()

    # Stability returns:
    # { "images": [ { "b64_json": "<base64>" } ] }
    img_b64 = data["images"][0]["b64_json"]
    img_bytes = base64.b64decode(img_b64)

    return _save_image_bytes(img_bytes, out_path)


# ---------------------------------------------------------------------
# 2. HuggingFace Inference API (Stable Diffusion or SDXL)
# ---------------------------------------------------------------------

def _generate_hf(prompt, out_path):
    api_key = os.getenv("HF_API_KEY")
    model = os.getenv("HF_SD_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

    if not api_key:
        raise ValueError("HF_API_KEY missing.")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {"inputs": prompt}

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()

    img_bytes = resp.content
    return _save_image_bytes(img_bytes, out_path)


# ---------------------------------------------------------------------
# 3. AUTOMATIC1111 Local Server
# ---------------------------------------------------------------------

def _generate_a1111(prompt, out_path):
    base_url = os.getenv("A1111_URL", "http://127.0.0.1:7860")

    payload = {"prompt": prompt}

    resp = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload)
    resp.raise_for_status()
    data = resp.json()

    img_bytes = base64.b64decode(data["images"][0])
    return _save_image_bytes(img_bytes, out_path)


# ---------------------------------------------------------------------
# 4. Replicate.com
# ---------------------------------------------------------------------

def _generate_replicate(prompt, out_path):
    api_token = os.getenv("REPLICATE_API_TOKEN")
    model = os.getenv("REPLICATE_MODEL", "stability-ai/sdxl")

    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN missing.")

    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "version": model,
        "input": {"prompt": prompt}
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    pred = resp.json()

    # Retrieve output URL
    image_url = pred["output"][0]
    img_bytes = requests.get(image_url).content
    return _save_image_bytes(img_bytes, out_path)


# ---------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------

def generate_sd(prompt, out_path="sd_image.png"):
    """
    Main Stable Diffusion entry point.
    Dispatches to one of several backends.

    Set:
      STABLE_DIFFUSION_PROVIDER = "stability" | "hf" | "a1111" | "replicate"
    """
    provider = os.getenv("STABLE_DIFFUSION_PROVIDER", "stability").lower()

    if provider == "stability":
        return _generate_stability(prompt, out_path)

    elif provider == "hf":
        return _generate_hf(prompt, out_path)

    elif provider == "a1111":
        return _generate_a1111(prompt, out_path)

    elif provider == "replicate":
        return _generate_replicate(prompt, out_path)

    else:
        raise ValueError(f"Unknown STABLE_DIFFUSION_PROVIDER={provider}")
