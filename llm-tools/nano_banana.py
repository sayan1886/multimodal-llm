# llm-tools/nano_banana.py

import os
import requests
import base64
import io
from PIL import Image


# ---------------------------------------------------------------------------
# INTERNAL: Save image bytes to project_root/output/
# ---------------------------------------------------------------------------

def _save_image_bytes(img_bytes, out_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    final_out = out_path if os.path.isabs(out_path) else os.path.join(output_dir, out_path)

    img = Image.open(io.BytesIO(img_bytes))
    img.save(final_out)
    return final_out


# ---------------------------------------------------------------------------
# MAIN PUBLIC API — Gemini Image Generation (Nano-Banana)
# ---------------------------------------------------------------------------

def generate_nano_banana(
    prompt,
    out_path="nano_image.png",
    timeout=40,
):
    """
    Nano-Banana = Gemini Image API wrapper.

    Required ENV:
      IMAGE_BACKEND="nano-banana"
      NANO_BANANA_API_KEY
      NANO_BANANA_MODEL   (default: gemini-2.5-flash-image)
    """

    # Must explicitly select nano-banana backend
    if os.getenv("IMAGE_BACKEND") != "nano-banana":
        raise RuntimeError("IMAGE_BACKEND must be set to 'nano-banana' to use Gemini image API.")

    api_key = os.getenv("NANO_BANANA_API_KEY")
    model_name = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")

    if not api_key:
        raise RuntimeError("Missing NANO_BANANA_API_KEY")

    # ✔ Correct Gemini image endpoint (per ai.google.dev)
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    # ✔ Correct request structure
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    # -----------------------------------------------------------------------
    # Send request
    # -----------------------------------------------------------------------
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Nano-Banana API request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Nano-Banana returned {resp.status_code}: {resp.text}")

    data = resp.json()

    # Expected response:
    # candidates[0].content.parts[0].inlineData.data (base64 image bytes)
    try:
        inline_data = (
            data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        )
    except Exception:
        raise RuntimeError(f"Gemini image response missing expected fields: {data}")

    img_bytes = base64.b64decode(inline_data)

    return _save_image_bytes(img_bytes, out_path)