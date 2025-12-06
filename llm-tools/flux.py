# llm-tools/flux.py

import os
import requests, base64, io
from PIL import Image

def generate_flux(prompt, out_path="flux_image.png", api_key=None, model=None, timeout=30):
    """Generate an image using the Flux/OpenAI Images API.

    - Validates `api_key` (env var `FLUX_API_KEY` or passed value).
    - Raises RuntimeError with a clear message on failure.
    """
    if api_key is None:
        api_key = os.getenv("FLUX_API_KEY")

    if not api_key or not str(api_key).strip():
        raise RuntimeError(
            "FLUX_API_KEY not set. Set the env var 'FLUX_API_KEY' or pass `api_key` to generate_flux.`"
        )

    endpoint = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Allow overriding model via parameter or FLUX_MODEL env var.
    if model is None:
        model = os.getenv("FLUX_MODEL", "gpt-image-1")

    body = {"model": model, "prompt": prompt, "size": "1024x1024"}

    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Request to Flux API failed: {e}")

    if resp.status_code != 200:
        # Surface the raw response for easier debugging
        text = resp.text
        raise RuntimeError(f"Flux API returned status {resp.status_code}: {text}")

    try:
        payload = resp.json()
    except Exception:
        raise RuntimeError(f"Flux API returned non-JSON response: {resp.text}")

    # Expecting payload['data'][0]['b64_json'] per API contract
    try:
        b64 = payload["data"][0]["b64_json"]
    except Exception:
        raise RuntimeError(f"Unexpected Flux response shape: {payload}")

    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(out_path)

    return out_path
