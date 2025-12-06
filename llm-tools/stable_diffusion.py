# llm-tools/stable_diffusion.py

import os
from PIL import Image, ImageDraw
import textwrap


def generate_sd(prompt, out_path="sd_image.png"):
    """Simple placeholder stable-diffusion generator.

    This creates a placeholder image with the prompt drawn on it.
    It's intended for offline/demo use when a real SD backend is not configured.
    """
    W, H = 1024, 1024
    bg = (30, 30, 40)
    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    # If prompt mentions robot and balloon, draw a simple illustrative robot holding a balloon.
    prompt_l = prompt.lower()
    if "robot" in prompt_l and "balloon" in prompt_l:
        # draw robot
        body_w, body_h = 300, 300
        body_x = (W - body_w) // 2
        body_y = (H - body_h) // 2
        draw.rectangle([body_x, body_y, body_x + body_w, body_y + body_h], fill=(200, 200, 220))
        # head
        head_r = 80
        head_x = W // 2
        head_y = body_y - head_r - 10
        draw.ellipse([head_x - head_r, head_y - head_r, head_x + head_r, head_y + head_r], fill=(200, 200, 220))
        # eyes
        eye_r = 12
        draw.ellipse([head_x - 30 - eye_r, head_y - 10 - eye_r, head_x - 30 + eye_r, head_y - 10 + eye_r], fill=(20, 20, 20))
        draw.ellipse([head_x + 30 - eye_r, head_y - 10 - eye_r, head_x + 30 + eye_r, head_y - 10 + eye_r], fill=(20, 20, 20))
        # antenna
        draw.line([head_x, head_y - head_r, head_x, head_y - head_r - 40], fill=(200, 200, 220), width=6)
        draw.ellipse([head_x - 8, head_y - head_r - 56, head_x + 8, head_y - head_r - 40], fill=(255, 80, 80))
        # arm and balloon string
        arm_x = body_x + body_w
        arm_y = body_y + 80
        draw.line([arm_x - 40, arm_y, arm_x + 120, arm_y - 200], fill=(200, 200, 220), width=12)
        # balloon
        balloon_x = arm_x + 120
        balloon_y = arm_y - 200
        draw.ellipse([balloon_x - 60, balloon_y - 80, balloon_x + 60, balloon_y + 80], fill=(220, 60, 80))
        # footer
        footer = "[placeholder stable-diffusion robot image]"
        draw.text((40, H - 40), footer, fill=(180, 180, 180))
    else:
        # Fallback: render the prompt text
        wrapped = textwrap.fill(prompt, width=40)
        margin = 40
        text_color = (240, 240, 240)
        draw.multiline_text((margin, margin), wrapped, fill=text_color)
        footer = "[placeholder stable-diffusion output]"
        draw.text((margin, H - 40), footer, fill=(180, 180, 180))

    # Save under project-level `output` directory to keep outputs centralized.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isabs(out_path):
        final_out = out_path
    else:
        final_out = os.path.join(output_dir, out_path)

    img.save(final_out)
    return final_out
