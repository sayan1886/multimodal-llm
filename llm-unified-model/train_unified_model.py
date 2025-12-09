# llm-unified-model/train_unified_model.py

import json
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import UnifiedImageTextDataset
from unified_model import UnifiedMultimodalLM


def load_pairs(json_path: str):
    """
    JSON must contain a list:
    [
       {"image_path": "...", "caption": "..."},
       ...
    ]
    """
    with open(json_path, "r") as f:
        return json.load(f)


def collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def train_unified(
    json_path="data/captions.json",
    batch_size=4,
    num_epochs=5,
    lr=5e-5,
    device=None,
    skip_train=True       # <==== NEW PARAM
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load dataset
    # -------------------------
    samples = load_pairs(json_path)
    dataset = UnifiedImageTextDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # -------------------------
    # Load or initialize model
    # -------------------------
    ckpt_path = "model_checkpoints/unified_model.pt"

    model = UnifiedMultimodalLM(prefix_tokens=16)

    # Must happen BEFORE loading checkpoint
    model.lm.resize_token_embeddings(len(dataset.tokenizer))

    # Load checkpoint if exists
    if os.path.exists(ckpt_path):
        print(f"Checkpoint detected → {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

        if skip_train:
            print("SKIP_TRAIN=True → Skipping training and returning loaded model.")
            model = model.to(device)
            model.eval()
            return model
        else:
            print("Resuming training...")
    else:
        print("No checkpoint found → Initializing new model")

    # -------------------------
    # If training is disabled and no ckpt exists
    # -------------------------
    if skip_train and not os.path.exists(ckpt_path):
        raise RuntimeError("SKIP_TRAIN=True but no checkpoint exists to load.")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(num_epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")

    # -------------------------
    # Save model
    # -------------------------
    os.makedirs("model_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved unified model → {ckpt_path}")

    return model


if __name__ == "__main__":
    train_unified()
