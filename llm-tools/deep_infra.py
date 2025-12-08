import os
import requests
import base64
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
API_KEY = os.getenv("DEEP_INFRA_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DEEP_INFRA_API_KEY in your environment variables.")

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# FUNCTION: Generate image
# ----------------------------------------------------------------------
def generate_image(prompt: str,
                   out_path: str = "deepinfra_image.png",
                   model: str = "stabilityai/stable-diffusion-2-1",
                   size: str = "512x512",
                   n: int = 1) -> str:
    """
    Generate an image using DeepInfra OpenAI-compatible API.

    Args:
        prompt (str): Text prompt for image generation.
        out_path (str): Filename to save the image.
        model (str): Model name on DeepInfra.
        size (str): Image size, e.g., "512x512".
        n (int): Number of images to generate.

    Returns:
        str: Path to saved image.
    """

    url = "https://api.deepinfra.com/v1/openai/images/generations"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "model": model,
        "size": size,
        "n": n
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    # Expect response: { "data": [ { "b64_json": "<base64>" }, ... ] }
    results = data.get("data")
    if not results:
        raise RuntimeError(f"Unexpected DeepInfra response: {data}")

    img_b64 = results[0].get("b64_json")
    if not img_b64:
        raise RuntimeError(f"No b64_json in response: {data}")

    img_bytes = base64.b64decode(img_b64)

    final_path = OUTPUT_DIR / out_path
    with open(final_path, "wb") as f:
        f.write(img_bytes)

    print(f"[INFO] Image saved to: {final_path}")
    return str(final_path)

# ----------------------------------------------------------------------
# TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prompt_text = input("Enter prompt for DeepInfra image: ")
    generate_image(prompt_text)
