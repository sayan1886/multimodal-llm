# llm-adapter/run_adapter.py

import os
import csv
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from train_adapter import train
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# Device selection
# -------------------------------
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# LOADERS
# -------------------------------
def load_flickr8k_captions(csv_path="./multimodal-dataset/flickr8k/captions.txt"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Flickr8k captions file not found: {csv_path}")

    # prefer lowercase 'images' then 'Images'
    images_dir = csv_path.parent / "images"
    if not images_dir.exists():
        images_dir = csv_path.parent / "Images"

    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            img_file = row[0]
            caption = row[1] if len(row) > 1 else ""
            img_path = str((images_dir / img_file.strip()).resolve()) if images_dir.exists() else str((csv_path.parent / img_file.strip()).resolve())
            examples.append({"image_path": img_path, "caption": caption.strip()})
    return examples


def load_json_dataset(json_path="./multimodal-dataset/captions.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Custom JSON dataset not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)

# -------------------------------
# Select dataset based on env var
# -------------------------------
def load_dataset():
    raw = os.environ.get("MM_DATASET", "")
    ds = raw.strip()

    print(f"MM_DATASET={ds or 'not set'}")

    # If MM_DATASET is a path to a directory, prefer that
    if ds:
        p = Path(ds)
        if p.exists() and p.is_dir():
            csv_path = p / "captions.txt"
            if csv_path.exists():
                print(f"→ Loading Flickr-like CSV dataset from {p}")
                return load_flickr8k_captions(str(csv_path)), "flickr"

        # Support known names
        low = ds.lower()
        if low in ("flickr8k", "flickr8k-subset", "flickr8k_subset"):
            # point to subset if requested
            if "subset" in low:
                csv_path = Path("./multimodal-dataset/flickr8k-subset/captions.txt")
            else:
                csv_path = Path("./multimodal-dataset/flickr8k/captions.txt")
            if csv_path.exists():
                print(f"→ Loading Flickr8k CSV dataset from {csv_path.parent}")
                return load_flickr8k_captions(str(csv_path)), "flickr"

    print("→ Loading custom JSON dataset (default)")
    return load_json_dataset(), "json"

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(img_path, image_size=224):
    img = Image.open(os.path.abspath(img_path)).convert("RGB")
    img = img.resize((image_size, image_size))
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(
        image_size, image_size, 3
    ).permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0)

# -------------------------------
# Inference
# -------------------------------
def generate_captions(model, examples, max_gen_len=20):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    model.eval()

    for ex in examples[:10]:  # limit for demo
        img_path = ex["image_path"]
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        pixel_values = preprocess_image(img_path).to(device)
        input_ids = torch.tensor([[bos]], device=device)
        generated = input_ids.clone()

        for _ in range(max_gen_len):
            with torch.no_grad():
                out = model(
                    pixel_values=pixel_values,
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated),
                )
                logits = out.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == eos:
                break

        caption = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nImage: {img_path}\nGenerated: {caption}")

        # ---- Show image with caption ----
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(caption)
        plt.show()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Load dataset (via env var)
    examples, ds_type = load_dataset()

    # JSON dataset requires JSON path for training
    if ds_type == "json":
        train_file = "./multimodal-dataset/captions.json"
    else:
        train_file = "./multimodal-dataset/flickr_captions_tmp.json"
        with open(train_file, "w") as f:
            json.dump(examples, f, indent=4)

    print("\n=== Training Adapter ===\n")
    model = train(
        data_json=train_file,
        batch_size=4,
        num_epochs=30,
        lr=1e-5
    )

    print("\n=== Generating Captions ===\n")
    generate_captions(model, examples)
