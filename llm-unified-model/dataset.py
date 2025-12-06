# llm-unified-model/dataset.py

import os
from PIL import Image
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import CLIPFeatureExtractor, GPT2TokenizerFast


class UnifiedImageTextDataset(Dataset):
    """
    Dataset for the unified multimodal LLM.
    Expects a list of:
        {"image_path": "...", "caption": "..."}
    """

    def __init__(self, examples: List[Dict], image_size=224, max_length=64):
        self.examples = examples
        self.image_size = image_size
        self.max_length = max_length

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        image = Image.open(ex["image_path"]).convert("RGB")
        image = self.transforms(image)
        clip_inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].squeeze(0)

        caption = ex["caption"]
        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
