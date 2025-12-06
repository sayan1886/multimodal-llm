# llm-adapter/dataset.py

import os
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2TokenizerFast, CLIPFeatureExtractor, CLIPVisionModel

class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs.
    Expects a list of dicts: {"image_path": ..., "caption": ...}
    """

    def __init__(
        self,
        examples: List[Dict],
        image_size: int = 224,
        max_length: int = 64,
    ):
        self.examples = examples
        self.image_size = image_size
        self.max_length = max_length

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img_path = ex["image_path"]
        caption = ex["caption"]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # shape: (3, H, W)

        # tokenize caption
        tokenized = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
