import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    Blip2QFormerConfig,
    Blip2QFormerModel,
    CLIPModel,
    CLIPProcessor,
)

class CombinedModel(nn.Module):
    """
    12-layer Q-Former architecture as designed in EmoEdit (Yang et al., 2024b) 
    and integrated in MERMAID (Yang et al., 2025).

    It fuses image features and target emotion embeddings via 76 learned query tokens:
    - Base: 12-layer Blip2QFormerModel (12 heads, hidden_size 768, encoder_hidden_size 768)
    - Query tokens: 76 learnable tokens of dim 768
    - Input: CLIP image embedding (1 token x 768) + CLIP emotion text embedding (1 token x 768)
    - Output: 77 fused emotion prompt tokens of dim 768 (matching SD 1.5 / InstructPix2Pix text latent space)
    """
    def __init__(self, num_query_tokens: int = 76, hidden_size: int = 768, ln: nn.Module = None):
        super(CombinedModel, self).__init__()
        qformer_config = Blip2QFormerConfig()
        qformer_config.encoder_hidden_size = hidden_size
        qformer_config.hidden_size = hidden_size
        qformer_config.num_hidden_layers = 12
        qformer_config.num_attention_heads = 12
        qformer_config.attention_probs_dropout_prob = 0.0

        self.model = Blip2QFormerModel(qformer_config)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.layer_norm = ln if ln is not None else nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, text_embeds: torch.Tensor, img_embeds: torch.Tensor) -> torch.Tensor:
        """
        Fuses text and image embeddings.
        Args:
            text_embeds: Tensor of shape [batch_size, 768] or [batch_size, 1, 768]
            img_embeds:  Tensor of shape [batch_size, 768] or [batch_size, 1, 768]
        Returns:
            normal_output: Tensor of shape [batch_size, 77, 768]
        """
        if img_embeds.dim() == 2:
            img_embeds = img_embeds.unsqueeze(1)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)

        batch_size = img_embeds.size(0)
        expand_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Concatenate 76 query tokens and 1 text embedding -> 77 tokens
        query = torch.cat([expand_query_tokens, text_embeds], dim=1)

        # Image attention mask
        image_attention_mask = torch.ones(
            img_embeds.size()[:-1], dtype=img_embeds.dtype, device=img_embeds.device
        )

        # Forward pass through Q-Former cross-attention
        query_outputs = self.model(
            query_embeds=query,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]

        # Layer normalization
        normal_output = self.layer_norm(query_outputs)
        return normal_output


def resize_image_to_512(img: Image.Image) -> Image.Image:
    """
    Resizes image maintaining aspect ratio with max dimension 512,
    ensuring width and height are multiples of 8 for Stable Diffusion VAE compatibility.
    """
    img = img.convert("RGB")
    width, height = img.size
    if width < height:
        new_width = 512
        new_height = int((height / width) * 512)
    else:
        new_height = 512
        new_width = int((width / height) * 512)
    new_width = max(64, (new_width // 8) * 8)
    new_height = max(64, (new_height // 8) * 8)
    return img.resize((new_width, new_height), Image.LANCZOS)
