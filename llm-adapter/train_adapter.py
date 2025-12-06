# module2_adapter/train_adapter.py

import os
import json
import torch
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from adapter import AdapterVisionLLM
from transformers import AdamW

def load_caption_data(json_path: str):
    """
    Load a JSON file expecting a list:
    [
      {"image_path": "...", "caption": "..."},
      ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def collate_fn(batch):
    # simple collate: default batching of tensors works since we fixed lengths
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}


def train(
    data_json: str,
    batch_size: int = 8,
    num_epochs: int = 3,
    lr: float = 5e-5,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    examples = load_caption_data(data_json)
    dataset = ImageCaptionDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = AdapterVisionLLM().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone().to(device)

            outputs = model(pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch+1} Step {step+1}/{len(dataloader)} Loss: {loss.item():.4f}")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg:.4f}")

    # Save the trained model
    os.makedirs("model_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "model_checkpoints/adapter_vision_llm.pt")
    print("Model saved at model_checkpoints/adapter_vision_llm.pt")


if __name__ == "__main__":
    # Example usage: assumes `data/captions.json` exists
    # Format: list of {"image_path": "...", "caption": "..."}
    train(data_json="data/captions.json", batch_size=4, num_epochs=5, lr=5e-5)
