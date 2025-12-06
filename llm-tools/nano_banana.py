# llm-tools/nano_banana.py

import os
import requests
import base64
import io
from PIL import Image
import textwrap


def _save_image_bytes(img_bytes, out_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isabs(out_path):
        final_out = out_path
    else:
        final_out = os.path.join(output_dir, out_path)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(final_out)
    return final_out


def generate_nano_banana(prompt, out_path="nano_image.png", api_key=None, endpoint=None, timeout=30):
    """Call a Nano-Banana style image endpoint or fall back to a placeholder image.

    Env vars:
    - `NANO_BANANA_ENDPOINT` : full URL to POST prompt JSON -> returns base64 image
    - `NANO_BANANA_API_KEY`  : optional Bearer token

    The function attempts to POST `{"prompt": prompt}` and then extract base64
    from common JSON fields (image_b64, b64, data[0].b64_json). If no endpoint is
    configured, it produces a placeholder image (offline-safe).
    """
    if endpoint is None:
        endpoint = os.getenv("NANO_BANANA_ENDPOINT")
    if api_key is None:
        api_key = os.getenv("NANO_BANANA_API_KEY")

    if endpoint:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.post(endpoint, json={"prompt": prompt}, headers=headers, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Nano-Banana request failed: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"Nano-Banana returned status {resp.status_code}: {resp.text}")
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError(f"Nano-Banana returned non-JSON: {resp.text}")

        # try common keys
        b64_candidates = [
            payload.get("image_b64"),
            payload.get("b64"),
        ]
        try:
            if "data" in payload and isinstance(payload["data"], list):
                b64_candidates.append(payload["data"][0].get("b64_json"))
        except Exception:
            pass

        b64 = None
        for c in b64_candidates:
            if c:
                b64 = c
                break
        if not b64:
            raise RuntimeError(f"Nano-Banana response missing expected base64 image field: {payload}")

        try:
            img_bytes = base64.b64decode(b64)
        except Exception:
            raise RuntimeError("Failed to decode base64 image from Nano-Banana response")

        return _save_image_bytes(img_bytes, out_path)

    # Fallback: placeholder image (draw simple robot+balloon for relevant prompts)
    from PIL import Image as PILImage, ImageDraw

    W, H = 1024, 1024
    bg = (20, 40, 30)
    img = PILImage.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    prompt_l = prompt.lower()
    if "robot" in prompt_l and "balloon" in prompt_l:
        # draw a simple robot with balloon (same as SD placeholder style)
        body_w, body_h = 300, 300
        body_x = (W - body_w) // 2
        body_y = (H - body_h) // 2
        draw.rectangle([body_x, body_y, body_x + body_w, body_y + body_h], fill=(200, 200, 220))
        head_r = 80
        head_x = W // 2
        head_y = body_y - head_r - 10
        draw.ellipse([head_x - head_r, head_y - head_r, head_x + head_r, head_y + head_r], fill=(200, 200, 220))
        eye_r = 12
        draw.ellipse([head_x - 30 - eye_r, head_y - 10 - eye_r, head_x - 30 + eye_r, head_y - 10 + eye_r], fill=(20, 20, 20))
        draw.ellipse([head_x + 30 - eye_r, head_y - 10 - eye_r, head_x + 30 + eye_r, head_y - 10 + eye_r], fill=(20, 20, 20))
        draw.line([head_x, head_y - head_r, head_x, head_y - head_r - 40], fill=(200, 200, 220), width=6)
        draw.ellipse([head_x - 8, head_y - head_r - 56, head_x + 8, head_y - head_r - 40], fill=(255, 80, 80))
        arm_x = body_x + body_w
        arm_y = body_y + 80
        draw.line([arm_x - 40, arm_y, arm_x + 120, arm_y - 200], fill=(200, 200, 220), width=12)
        balloon_x = arm_x + 120
        balloon_y = arm_y - 200
        draw.ellipse([balloon_x - 60, balloon_y - 80, balloon_x + 60, balloon_y + 80], fill=(220, 60, 80))
        footer = "[nano-banana placeholder robot image]"
        draw.text((40, H - 40), footer, fill=(180, 180, 180))
    else:
        wrapped = textwrap.fill(prompt, width=40)
        margin = 40
        text_color = (240, 240, 240)
        draw.multiline_text((margin, margin), wrapped, fill=text_color)
        footer = "[nano-banana placeholder output]"
        draw.text((margin, H - 40), footer, fill=(180, 180, 180))

    # Save placeholder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    final_out = os.path.join(output_dir, out_path)
    img.save(final_out)
    return final_out
