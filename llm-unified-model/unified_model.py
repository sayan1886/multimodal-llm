# llm-unified-model/unified_model.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2TokenizerFast

class UnifiedMultimodalLM(nn.Module):
    def __init__(self, prefix_tokens=16, proj_dim=512):
        super().__init__()

        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        vis_dim = self.vision.config.hidden_size
        lm_dim = self.lm.config.n_embd

        self.prefix_tokens = prefix_tokens

        self.proj = nn.Sequential(
            nn.Linear(vis_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, prefix_tokens * lm_dim)
        )

        self.ln = nn.LayerNorm(lm_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        v = self.vision(pixel_values).last_hidden_state.mean(dim=1)
        p = self.ln(self.proj(v).view(-1, self.prefix_tokens, self.lm.config.n_embd))

        tok = self.lm.transformer.wte(input_ids)
        fused = torch.cat([p, tok], dim=1)

        labels = torch.cat([torch.full((labels.size(0), self.prefix_tokens), -100), labels], dim=1)

        return self.lm(inputs_embeds=fused, attention_mask=None, labels=labels)
