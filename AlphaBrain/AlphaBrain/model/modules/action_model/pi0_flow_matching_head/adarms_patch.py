"""
AdaRMS Normalization Patch for π₀.5

Monkey-patches transformers' GemmaRMSNorm and GemmaDecoderLayer to support
adaptive RMS normalization (adaRMS) required by π₀.5's action expert.

_orig_model_forward is exported for use in KV cache inference (sample_actions).

In π0.5, the action expert's normalization layers use a conditioning signal
(the flow matching timestep embedding) to adaptively scale and shift the
normalized hidden states, plus a gating mechanism for residual connections.

This patch is version-agnostic - it works with any transformers version that
has GemmaRMSNorm, without replacing entire files.

Usage:
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.adarms_patch import patch_gemma_for_adarms
    patch_gemma_for_adarms()  # Call once before creating models
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

# Module-level reference to original GemmaModel.forward (set by patch_gemma_for_adarms)
_orig_model_forward = None

logger = logging.getLogger(__name__)

_PATCHED = False


class AdaRMSNorm(nn.Module):
    """
    Adaptive RMS Normalization.

    When cond_dim is None: behaves like standard GemmaRMSNorm.
    When cond_dim is set: uses a dense layer to produce (scale, shift, gate)
    from the conditioning signal, enabling adaptive normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed = self._norm(x)

        if cond is None or self.dense is None:
            if hasattr(self, 'weight'):
                normed = normed * (1.0 + self.weight.float())
            # If no weight and no cond, just return normed (identity scale)
            return normed.to(dtype), None

        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()

        return normed.to(dtype), gate.to(dtype)


def _gated_residual(x, y, gate):
    """Gated residual connection used in adaRMS layers."""
    if gate is None:
        return x + y
    return x + gate * y


def patch_gemma_for_adarms():
    """
    Monkey-patch transformers.models.gemma to support adaRMS.

    This patches:
    1. GemmaRMSNorm → AdaRMSNorm (supports cond parameter)
    2. GemmaDecoderLayer.forward → accepts adarms_cond
    3. GemmaModel.forward → passes adarms_cond through layers
    4. Adds _gated_residual to modeling_gemma

    Safe to call multiple times (idempotent).
    """
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.gemma import modeling_gemma

    # Skip patching if source already has adaRMS support (e.g. AlphaBrain_pi05 env)
    import inspect
    _fwd_sig = inspect.signature(modeling_gemma.GemmaRMSNorm.forward)
    if 'cond' in _fwd_sig.parameters:
        print('[adarms_patch] GemmaRMSNorm already has adaRMS (source-patched env), skipping monkey-patch', flush=True)
        _PATCHED = True
        return

    # 1. Replace GemmaRMSNorm
    modeling_gemma.GemmaRMSNorm = AdaRMSNorm

    # 2. Add _gated_residual
    modeling_gemma._gated_residual = _gated_residual

    # 3. Patch GemmaDecoderLayer.__init__ to use cond_dim from config
    _orig_decoder_init = modeling_gemma.GemmaDecoderLayer.__init__

    def _patched_decoder_init(self, config, layer_idx):
        _orig_decoder_init(self, config, layer_idx)
        # Replace norms with AdaRMS if config has adarms settings
        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None
        if cond_dim is not None:
            self.input_layernorm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
            self.post_attention_layernorm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    modeling_gemma.GemmaDecoderLayer.__init__ = _patched_decoder_init

    # 4. Patch GemmaDecoderLayer.forward to accept and use adarms_cond
    _orig_decoder_forward = modeling_gemma.GemmaDecoderLayer.forward

    def _patched_decoder_forward(self, hidden_states, attention_mask=None, position_ids=None,
                                  past_key_values=None, past_key_value=None,
                                  output_attentions=False, use_cache=False,
                                  cache_position=None, position_embeddings=None,
                                  adarms_cond=None, **kwargs):
        # Compat: accept both past_key_value (old) and past_key_values (new)
        _past_kv = past_key_values if past_key_values is not None else past_key_value

        residual = hidden_states

        # Input layernorm (may return gate for adaRMS)
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)

        # Self attention
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Gated residual
        hidden_states = _gated_residual(residual, attn_output, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)

        # Return single tensor to match transformers >= 4.57 GemmaDecoderLayer signature
        return hidden_states

    modeling_gemma.GemmaDecoderLayer.forward = _patched_decoder_forward

    # 5. Patch GemmaModel to support adarms_cond in forward
    global _orig_model_forward
    _orig_model_cls = modeling_gemma.GemmaModel
    _orig_model_forward = _orig_model_cls.forward

    def _patched_model_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                                past_key_values=None, inputs_embeds=None, use_cache=None,
                                output_attentions=None, output_hidden_states=None,
                                return_dict=None, cache_position=None, adarms_cond=None, **kwargs):
        # Manual layer iteration -- bypasses _orig_model_forward entirely.
        #
        # KEY FIX: transformers 4.57.0 applies hidden_states *= sqrt(hidden_size) normalization
        # in GemmaModel.forward, but in 4.53.2 (the training env) this line was COMMENTED OUT.
        # Bypassing _orig_model_forward and NOT applying that normalizer matches training
        # behavior and fixes oscillating denoising in multi-step inference.

        from transformers.modeling_outputs import BaseModelOutputWithPast

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # DO NOT apply the sqrt(hidden_size) normalizer here!
        # In transformers 4.53.2 (training env), that line was commented out.
        # Applying it (as 4.57.0 does) causes ~32x scaling -> oscillating denoising.

        # Create DynamicCache when use_cache=True but no cache provided yet
        # (this is the VLM prefix forward pass that needs to BUILD the KV cache)
        if use_cache and past_key_values is None:
            try:
                from transformers import DynamicCache
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                from transformers import DynamicCache
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
            )

        # Apply final adaRMS norm with conditioning
        if adarms_cond is not None and hasattr(self.norm, 'dense') and self.norm.dense is not None:
            hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)
        else:
            result_norm = self.norm(hidden_states)
            if isinstance(result_norm, tuple):
                hidden_states = result_norm[0]
            else:
                hidden_states = result_norm

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    _orig_model_cls.forward = _patched_model_forward

    # 6. Patch final norm in GemmaModel to support adaRMS
    _orig_model_init = _orig_model_cls.__init__

    def _patched_model_init(self, config):
        _orig_model_init(self, config)
        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None
        if cond_dim is not None:
            self.norm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)


    _orig_model_cls.__init__ = _patched_model_init

    # Patch _init_weights on the PreTrainedModel base class
    # This is critical: GemmaModel.post_init() calls _init_weights during __init__
    for cls_name in ['GemmaPreTrainedModel', 'GemmaForCausalLM']:
        cls = getattr(modeling_gemma, cls_name, None)
        if cls and hasattr(cls, '_init_weights'):
            _orig_iw = cls._init_weights
            def _safe_init_weights(self, module, _orig=_orig_iw):
                if isinstance(module, AdaRMSNorm):
                    if module.dense is not None:
                        nn.init.zeros_(module.dense.weight)
                        if module.dense.bias is not None:
                            nn.init.zeros_(module.dense.bias)
                    elif hasattr(module, 'weight'):
                        module.weight.data.zero_()
                    return
                if 'RMSNorm' in module.__class__.__name__ and not hasattr(module, 'weight'):
                    return
                _orig(self, module)
            cls._init_weights = _safe_init_weights

    _PATCHED = True

