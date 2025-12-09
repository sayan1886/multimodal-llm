# llm-unified-model/unified_model.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.modeling_outputs import CausalLMOutput

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

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):

        # 1. Vision → projected prefix
        v = self.vision(pixel_values).last_hidden_state.mean(dim=1)
        p = self.ln(self.proj(v).view(-1, self.prefix_tokens, self.lm.config.n_embd))

        # 2. Token embeddings
        tok = self.lm.transformer.wte(input_ids)

        # 3. Concatenate prefix + tokens
        fused = torch.cat([p, tok], dim=1)

        # Adjust attention mask
        if attention_mask is not None:
            B = input_ids.size(0)
            prefix_mask = torch.ones((B, self.prefix_tokens), device=fused.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # -----------------------------
        # TRAINING MODE (labels provided)
        # -----------------------------
        if labels is not None:
            prefix_mask = torch.full(
                (labels.size(0), self.prefix_tokens),
                -100,
                device=labels.device
            )
            labels = torch.cat([prefix_mask, labels], dim=1)

            outputs = self.lm(
                inputs_embeds=fused,
                attention_mask=attention_mask,
                labels=labels
            )

            # Always return a structure with .logits
            return CausalLMOutput(
                loss=outputs.loss,
                logits=outputs.logits
            )

        # -----------------------------
        # INFERENCE MODE
        # -----------------------------
        else:
            outputs = self.lm(
                inputs_embeds=fused,
                attention_mask=attention_mask,
                use_cache=True
            )

            # No loss because labels=None
            return CausalLMOutput(
                loss=None,
                logits=outputs.logits
            )
