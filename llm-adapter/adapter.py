# llm-adapter/adapter.py

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2TokenizerFast

class AdapterVisionLLM(nn.Module):
    """
    Combines a CLIP vision encoder with a GPT-2 language model using a learnable adapter
    to fuse visual and text embeddings for image captioning or multimodal generation.
    """

    def __init__(self, adapter_dim=768):
        super().__init__()

        # Vision Encoder
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Text LLM
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.lm.resize_token_embeddings(len(self.tokenizer))

        # Adapter: maps vision embedding to LM embedding space
        vis_dim = self.vision.config.hidden_size
        lm_dim = self.lm.config.n_embd

        self.adapter = nn.Sequential(
            nn.Linear(vis_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, lm_dim)
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Encode image
        vis_feat = self.vision(pixel_values).last_hidden_state.mean(dim=1)  # B x vis_dim
        adapted = self.adapter(vis_feat).unsqueeze(1)  # B x 1 x lm_dim

        # Token embeddings
        tok_emb = self.lm.transformer.wte(input_ids)  # B x seq_len x lm_dim

        # Concatenate visual token to text tokens
        fused = torch.cat([adapted, tok_emb], dim=1)  # B x (1 + seq_len) x lm_dim

        # Adjust attention mask
        if attention_mask is not None:
            vis_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            fused_mask = torch.cat([vis_mask, attention_mask], dim=1)
        else:
            fused_mask = None

        # Adjust labels to ignore the visual token in loss
        if labels is not None:
            ignore_index = -100  # GPT2 ignores -100 tokens
            # Prepend ignore token for visual embedding
            fused_labels = torch.cat([torch.full((labels.size(0), 1), ignore_index, device=labels.device), labels], dim=1)
        else:
            fused_labels = None

        return self.lm(
            inputs_embeds=fused,
            attention_mask=fused_mask,
            labels=fused_labels
        )
