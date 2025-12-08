# llm-tools/freepik_api.py

import os
import requests
from pathlib import Path
import time

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
API_KEY = os.getenv("FREEPIK_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set FREEPIK_API_KEY in your environment variables.")

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# FUNCTION: Generate Image
# ----------------------------------------------------------------------
def generate_image(
    prompt: str,
    out_path: str = "freepik_image.png",
    aspect_ratio: str = "square_1_1",
    seed: int = None,
    effects: dict = None,
    colors: list = None,
    webhook_url: str = None,
    poll_interval: int = 5,
    timeout: int = 300
) -> str:
    """
    Generate an image using Freepik Text-to-Image API (Flux-Dev).

    Works asynchronously:
      1. POST to create a task → returns task_id
      2. GET task_id until status is COMPLETED → get image URL(s)
      3. Download image from URL and save locally

    Args:
        prompt (str): Text prompt for image generation.
        out_path (str): Filename to save the image.
        aspect_ratio (str): Aspect ratio, e.g., "square_1_1"
        seed (int): Optional seed for reproducibility.
        effects (dict): Optional styling effects.
        colors (list): Optional color weights [{"color": "#FF0000", "weight": 0.5}]
        webhook_url (str): Optional webhook URL for async generation.
        poll_interval (int): Seconds between polling attempts.
        timeout (int): Max seconds to wait for completion.

    Returns:
        str: Path to saved image
    """

    post_url = "https://api.freepik.com/v1/ai/text-to-image/flux-dev"
    headers = {
        "Content-Type": "application/json",
        "x-freepik-api-key": API_KEY
    }

    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio
    }

    if effects:
        payload["styling"] = {"effects": effects}
    if colors:
        payload.setdefault("styling", {}).setdefault("colors", colors)
    if seed:
        payload["seed"] = seed
    if webhook_url:
        payload["webhook_url"] = webhook_url

    # Step 1: Create task
    resp = requests.post(post_url, headers=headers, json=payload)
    resp.raise_for_status()
    task_id = resp.json()["data"]["task_id"]

    # Step 2: Poll for completion
    get_url = f"{post_url}/{task_id}"
    start_time = time.time()
    while True:
        r = requests.get(get_url, headers={"x-freepik-api-key": API_KEY})
        r.raise_for_status()
        data = r.json()["data"]
        status = data["status"]

        if status == "COMPLETED":
            img_url = data["generated"][0]
            # Download the image
            img_resp = requests.get(img_url)
            img_resp.raise_for_status()
            final_path = OUTPUT_DIR / out_path
            with open(final_path, "wb") as f:
                f.write(img_resp.content)
            print(f"[INFO] Freepik image saved to: {final_path}")
            return str(final_path)
        elif status == "FAILED":
            raise RuntimeError(f"Freepik generation failed: {data}")
        elif time.time() - start_time > timeout:
            raise RuntimeError("Freepik generation timed out.")
        else:
            time.sleep(poll_interval)


# ----------------------------------------------------------------------
# TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prompt_text = input("Enter prompt for Freepik image: ")
    generate_image(prompt_text)
