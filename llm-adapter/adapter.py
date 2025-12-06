# llm-adapter/adapter.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2TokenizerFast

class AdapterVisionLLM(nn.Module):
    def __init__(self, adapter_dim=768):
        super().__init__()

        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        vis_dim = self.vision.config.hidden_size
        lm_dim = self.lm.config.n_embd

        self.adapter = nn.Sequential(
            nn.Linear(vis_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, lm_dim)
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        vis = self.vision(pixel_values).last_hidden_state.mean(dim=1)
        adapted = self.adapter(vis).unsqueeze(1)
        tok_emb = self.lm.transformer.wte(input_ids)

        fused = torch.cat([adapted, tok_emb], dim=1)

        return self.lm(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            labels=labels
        )
