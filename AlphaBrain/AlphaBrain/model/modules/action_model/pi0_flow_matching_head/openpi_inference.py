"""
OpenPI-compatible inference for AlphaBrain's PaliGemmaOFT.

Wraps AlphaBrain's VLM + Action Expert into openpi's PaliGemmaWithExpertModel
structure, then uses openpi's proven KV cache inference path.

This is the bridge between AlphaBrain's modular architecture and openpi's
inference implementation.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .openpi_gemma import PaliGemmaWithExpertModel


def create_sinusoidal_pos_embedding(timestep, dim, min_period=4e-3, max_period=4.0, device=None):
    """Sinusoidal position embedding for flow matching timestep."""
    if timestep.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    half = dim // 2
    freqs = torch.exp(-math.log(min_period / max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half)
    args = timestep[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def make_att_2d_masks(pad_masks, att_masks):
    """Build 2D attention masks from padding masks and autoregressive masks."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def build_openpi_wrapper(vlm_language_model, vlm_interface, action_expert_model, flow_matching_head):
    """
    Build an openpi-compatible PaliGemmaWithExpertModel from AlphaBrain components.
    
    This creates a thin wrapper that makes AlphaBrain's modules look like
    openpi's PaliGemmaWithExpertModel for inference purposes.
    """
    class AlphaBrainAsOpenPI(nn.Module):
        """Adapter that makes AlphaBrain look like openpi's model structure."""
        
        def __init__(self):
            super().__init__()
            # Create paligemma-like structure
            self.paligemma = type('PaliGemma', (), {
                'model': vlm_interface.model,
                'language_model': vlm_language_model,
                'config': type('Config', (), {
                    'text_config': vlm_language_model.config,
                })(),
            })()
            
            self.gemma_expert = type('GemmaExpert', (), {
                'model': action_expert_model,
            })()
            
        def embed_image(self, image):
            return vlm_interface.model.get_image_features(image)
        
        def embed_language_tokens(self, tokens):
            return vlm_interface.model.embed_tokens(tokens)
        
        def to_bfloat16_for_selected_params(self, precision="bfloat16"):
            """Match openpi's selective precision control."""
            if precision == "bfloat16":
                self.to(dtype=torch.bfloat16)
            elif precision == "float32":
                self.to(dtype=torch.float32)
                return
            
            params_to_keep_float32 = [
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                "vision_tower.vision_model.embeddings.patch_embedding.bias",
                "vision_tower.vision_model.embeddings.position_embedding.weight",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
        
        def forward(self, attention_mask, position_ids, past_key_values, 
                    inputs_embeds, use_cache=None, adarms_cond=None):
            """Route to appropriate forward path (same as openpi's gemma_pytorch.py)."""
            if adarms_cond is None:
                adarms_cond = [None, None]
            
            if inputs_embeds[1] is None:
                # Prefix-only: run through VLM language model
                prefix_output = self.paligemma.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0],
                )
                return None, prefix_output.past_key_values
                
            elif inputs_embeds[0] is None:
                # Suffix-only: run through action expert with KV cache
                suffix_output = self.gemma_expert.model.forward(
                    inputs_embeds=inputs_embeds[1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1],
                )
                return [None, suffix_output.last_hidden_state], None
            else:
                raise NotImplementedError("Joint forward through wrapper not implemented (use _shared_forward for training)")
    
    return AlphaBrainAsOpenPI()


def sample_actions_openpi(
    wrapper,
    flow_matching_head,
    prefix_embs,
    prefix_pad_masks,
    prefix_att_masks,
    device,
    noise=None,
    num_steps=10,
    state=None,
):
    """
    OpenPI-compatible sample_actions using KV cache.
    
    This is a direct port of PI0Pytorch.sample_actions + denoise_step,
    adapted to work with the AlphaBrainAsOpenPI wrapper.
    """
    bsize = prefix_pad_masks.shape[0]
    action_horizon = flow_matching_head.action_horizon
    action_dim = flow_matching_head.action_dim
    
    if noise is None:
        noise = torch.randn(bsize, action_horizon, action_dim, dtype=torch.float32, device=device)

    # Use autocast for mixed precision compatibility (torch 2.6 requires matching dtypes for F.linear)
    return _sample_actions_impl(
        wrapper, flow_matching_head, prefix_embs, prefix_pad_masks, prefix_att_masks,
        device, noise, num_steps, state)

def _sample_actions_impl(
    wrapper, flow_matching_head, prefix_embs, prefix_pad_masks, prefix_att_masks,
    device, noise, num_steps, state):
    bsize = prefix_pad_masks.shape[0]
    action_horizon = flow_matching_head.action_horizon
    action_dim = flow_matching_head.action_dim
    
    if noise is None:
        noise = torch.randn(bsize, action_horizon, action_dim, dtype=torch.float32, device=device)

    # Step 1: Compute prefix KV cache
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_4d = prefix_att_2d[:, None, :, :]
    prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)
    
    wrapper.paligemma.language_model.config._attn_implementation = "eager"
    
    _, past_key_values = wrapper.forward(
        attention_mask=prefix_att_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    
    # Step 2: Iterative denoising
    dt = -1.0 / num_steps
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    while time >= -dt_t / 2:
        expanded_time = time.expand(bsize)
        
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = flow_matching_head.embed_suffix(
            state, x_t, expanded_time
        )
        
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        
        # Build attention mask for suffix (attending to prefix KV + self)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = full_att_2d[:, None, :, :]
        full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)
        
        # Position IDs for suffix
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # Run suffix through expert with KV cache
        wrapper.gemma_expert.model.config._attn_implementation = "eager"
        
        outputs_embeds, _ = wrapper.forward(
            attention_mask=full_att_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = flow_matching_head.action_out_proj(suffix_out)
        
        x_t = x_t + dt_t * v_t
        time = time + dt_t
    
    return x_t
