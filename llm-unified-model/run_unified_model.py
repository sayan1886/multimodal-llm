# llm-unified-model/run_unified_model.py

import csv
import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from pathlib import Path
from train_unified_model import train_unified
from torchvision import transforms

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("./model_checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 5
LR = 5e-5
MAX_EXAMPLES = 5  # for demo

# ----------------------------
# Load dataset
# ----------------------------
def load_flickr8k_captions(csv_path="./multimodal-dataset/flickr8k/captions.txt"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Flickr8k captions file not found: {csv_path}")

    images_dir = next((csv_path.parent / d for d in ["images", "Images"] if (csv_path.parent / d).exists()), None)

    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            img_file = row[0]
            caption = row[1] if len(row) > 1 else ""
            img_path = images_dir / img_file.strip() if images_dir else csv_path.parent / img_file.strip()
            examples.append({"image_path": str(img_path.resolve()), "caption": caption.strip()})
    return examples

def load_json_dataset(json_path="./multimodal-dataset/captions.json"):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Custom JSON dataset not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def load_dataset(type=None):    
    if type == "test":
        print("Loading test dataset")
        return load_json_dataset("./multimodal-dataset/captions.json"), "json"
    
    raw = os.environ.get("MM_DATASET", "").strip()
    print(f"MM_DATASET={raw or 'not set'}")

    if raw:
        p = Path(raw)
        if p.exists() and p.is_dir():
            csv_path = p / "captions.txt"
            if csv_path.exists():
                print(f"→ Loading Flickr-like CSV dataset from {p}")
                return load_flickr8k_captions(str(csv_path)), "flickr"

        low = raw.lower()
        if low in ("flickr8k", "flickr8k-subset", "flickr8k_subset"):
            csv_path = Path("./multimodal-dataset/flickr8k-subset/captions.txt") if "subset" in low else Path("./multimodal-dataset/flickr8k/captions.txt")
            if csv_path.exists():
                print(f"Loading Flickr8k CSV dataset from {csv_path.parent}")
                return load_flickr8k_captions(str(csv_path)), "flickr"

    print("Loading custom JSON dataset (default)")
    return load_json_dataset(), "json"

# -------------------------------
# Preprocess image
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(img_path):
    img = Image.open(Path(img_path).resolve()).convert("RGB")
    return preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

# -------------------------------
# Inference
# -------------------------------
def generate_captions(model, examples, output_dir="output", max_gen_len=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # avoid warnings

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    device = next(model.parameters()).device
    model.eval()
    results = []

    for ex in examples[:MAX_EXAMPLES]:
        img_path = ex["image_path"]
        if not Path(img_path).exists():
            print(f"Missing image: {img_path}")
            continue

        try:
            pixel_values = preprocess_image(img_path).to(device)
            input_ids = torch.tensor([[bos]], device=device)
            generated = input_ids.clone()

            for _ in range(max_gen_len):
                with torch.no_grad():
                    out = model(pixel_values=pixel_values,
                                input_ids=generated,
                                attention_mask=torch.ones_like(generated))
                    logits = out.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == eos:
                    break

            caption = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Image: {img_path}\nGenerated: {caption}")

            results.append({"image_path": img_path, "caption": caption})

            # Save image with caption
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(caption)
            plt.savefig(output_dir / f"{Path(img_path).stem}_caption.png")
            plt.close()

        except Exception as e:
            print(f"Error generating caption for {img_path}: {e}")
            continue

    # Save JSON results
    json_path = output_dir / "captions_generated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nSaved {len(results)} captions to {json_path}")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    examples, ds_type = load_dataset()

    if ds_type == "json":
        train_file = "./multimodal-dataset/captions.json"
    else:
        train_file = "./multimodal-dataset/flickr_captions_tmp.json"
        with open(train_file, "w") as f:
            json.dump(examples, f, indent=4)
    
    print("\n=== Training Unified Model ===\n")
    checkpoint_dir = Path(__file__).parent / "model_checkpoints"
    print(f"Checkpoint directory: {checkpoint_dir}")

    model = train_unified(
        json_path=str(train_file),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE
    )

    print("\nTraining finished. Model is ready.")

    ckpt_path = OUTPUT_DIR / "unified_model_demo.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved unified model → {ckpt_path}")

    print("\n=== Generating Captions ===\n")
    test_examples, _ = load_dataset(type="test")
    generate_captions(model, test_examples)
