# llm-unified-model/train_unified_model.py

import json
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
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    samples = load_pairs(json_path)
    dataset = UnifiedImageTextDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Model
    model = UnifiedMultimodalLM(prefix_tokens=16).to(device)
    # Important: resize GPT2 embeddings if tokenizer added new tokens
    model.lm.resize_token_embeddings(len(dataset.tokenizer))
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

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

    # Save model
    torch.save(model.state_dict(), "model_checkpoints/unified_model.pt")
    print("Saved unified model → model_checkpoints/unified_model.pt")


if __name__ == "__main__":
    train_unified()
