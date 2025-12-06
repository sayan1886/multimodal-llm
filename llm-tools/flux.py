# llm-tools/flux.py

import requests, base64, io
from PIL import Image

def generate_flux(prompt, out_path="flux_image.png"):
    endpoint = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer YOUR_API_KEY"}

    body = {"model": "flux", "prompt": prompt, "size": "1024x1024"}
    resp = requests.post(endpoint, headers=headers, json=body)
    
    img_bytes = base64.b64decode(resp.json()["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes))
    img.save(out_path)

    return out_path
