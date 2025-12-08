# llm-adapter/dataset.py

import os
from typing import List, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2TokenizerFast, CLIPFeatureExtractor

class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs.
    Expects a list of dicts: {"image_path": ..., "caption": ...}
    """

    def __init__(self, examples: List[Dict], image_size: int = 224, max_length: int = 64):
        self.examples = examples
        self.image_size = image_size
        self.max_length = max_length

        # CLIP feature extractor (optional, can be used for future)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        # GPT2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Define transform once
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),  # scales 0-255 to 0-1
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img_path = ex["image_path"]
        caption = ex["caption"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found.")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)  # shape: (3, H, W)

        # Tokenize caption
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
            "labels": input_ids.clone()
        }
