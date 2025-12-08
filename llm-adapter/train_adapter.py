# llm-adapter/train_adapter.py

import os
import json
import torch
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from adapter import AdapterVisionLLM
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

def load_caption_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def train(
    data_json: str,
    batch_size: int = 8,
    num_epochs: int = 5,
    lr: float = 5e-5,
    patience: int = 3,
    device: str = None,
):
    # ----------------------------
    # Device selection (MPS, CUDA)
    # ----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # ----------------------------
    # Load dataset
    # ----------------------------
    examples = load_caption_data(data_json)
    dataset = ImageCaptionDataset(examples)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    # ----------------------------
    # Model
    # ----------------------------
    model = AdapterVisionLLM().to(device)

    ckpt_path = "model_checkpoints/adapter_vision_llm.pt"
    os.makedirs("model_checkpoints", exist_ok=True)

    # Resume if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    optimizer = AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler with warmup (10% warmup)
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ----------------------------
    # Training loop with early stop
    # ----------------------------
    best_loss = float("inf")
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # MPS-friendly autocast
            with torch.autocast(device_type=device.type, enabled=(device.type != "cpu")):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            loss.backward()
            
                        # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} — Avg Loss: {avg_loss:.4f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0

            # Save checkpoint on improvement
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Checkpoint saved at {ckpt_path}")

        else:
            wait += 1
            print(f"No improvement. Patience {wait}/{patience}")

            if wait >= patience:
                print("⛔ Early stopping triggered.")
                break

    print("Training finished.")
    return model

