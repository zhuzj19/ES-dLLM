# Modified from Dream Huggingface library

import torch
from typing import Optional, List, Tuple, Union, Dict
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput
from transformers.utils import logging
import early_skipping

logger = logging.get_logger(__name__)

def DreamModel_forward(self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    data_caches: Optional[List[Dict]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None, # not used
    labels: Optional[torch.LongTensor] = None, # not used
    use_cache: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None, # not used
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, MaskedLMOutput]:

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        data_caches=data_caches,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return MaskedLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def DreamBaseModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    data_caches: Optional[List[Dict]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutput]:
    assert self.training == False, "Training is not supported."

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache:
        assert data_caches is not None, "data_caches must be provided when use_cache=True"
        assert len(data_caches) == len(self.layers), "data_caches must match the number of layers"

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    q_pos, k_pos = early_skipping.get_qk_position(data_caches[0], position_ids=position_ids)
    position_embeddings = self.rotary_emb(hidden_states, q_pos), self.rotary_emb(hidden_states, k_pos)

    for decoder_layer, data_cache in zip(self.layers, data_caches):

        if self.gradient_checkpointing and self.training:
            layer_output, position_ids, attention_mask, position_embeddings = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                data_cache,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_output, position_ids, attention_mask, position_embeddings = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                data_cache=data_cache,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_output

    hidden_states = self.norm(hidden_states)

    if not return_dict:
        return tuple(v for v in [hidden_states] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
    )

def DreamDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    data_cache: Optional[Dict] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    if data_cache['record_time']:
        data_cache['start_proj_event'].record()
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, position_ids, attention_mask, position_embeddings, selected_indices = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        data_cache=data_cache,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    if selected_indices is not None:
        residual = residual.take_along_dim(selected_indices[..., None], 1)
    hidden_states = residual + hidden_states

    # Fully Connected
    if data_cache['record_time']:
        data_cache['start_ffn_event'].record()
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    if data_cache['record_time']:
        data_cache['end_ffn_event'].record()
    
    if data_cache['ESdLLM_mode'] == "HiddenState":
        if data_cache['record_time']:
            data_cache['start_importance_score'].record()
        hidden_states, position_ids, attention_mask, selected_indices = early_skipping.skip_in_hidden_state(data_cache, hidden_states, position_ids, attention_mask, importance_score_alpha=data_cache['importance_score_alpha'])
        if data_cache['record_time']:
            data_cache['end_importance_score'].record()
        if selected_indices is not None:
            position_embeddings = (position_embeddings[0][0].take_along_dim(selected_indices[..., None], 1),  position_embeddings[0][1].take_along_dim(selected_indices[..., None], 1)), (position_embeddings[1][0].take_along_dim(selected_indices[..., None], 1),  position_embeddings[1][1].take_along_dim(selected_indices[..., None], 1))
        early_skipping.update_hidden_state(data_cache, hidden_states, position_ids)
    elif data_cache['statistics']:
        early_skipping.update_hidden_state(data_cache, hidden_states, position_ids)

    output = hidden_states

    return output, position_ids, attention_mask, position_embeddings

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    tensor_embed = (tensor * cos) + (rotate_half(tensor) * sin)
    return tensor_embed

def DreamSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    data_cache: Optional[Dict] = None,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    if data_cache['record_time']:
        data_cache['end_proj_event'].record()
    
    if data_cache['record_time']:
        data_cache['start_attn_event'].record()

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        raise ValueError("`position_embeddings` must be provided to the attention layer.")
    else:
        (q_cos, q_sin), (k_cos, k_sin) = position_embeddings
    query_states = apply_rotary_pos_emb(query_states, q_cos, q_sin)
    key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin)

    selected_indices = None
    if data_cache['ESdLLM_mode'] in ["Key", "Value", "Query"]:
        query_states, key_states, value_states, position_ids, attention_mask, selected_indices = early_skipping.skip_in_qkv(data_cache, query_states, key_states, value_states, position_ids, attention_mask, importance_score_alpha=data_cache['importance_score_alpha'], absdiff_cfg=data_cache['ESdLLM_mode'])
        if selected_indices is not None:
            position_embeddings = (position_embeddings[0][0].take_along_dim(selected_indices[..., None], 1),  position_embeddings[0][1].take_along_dim(selected_indices[..., None], 1)), (position_embeddings[1][0].take_along_dim(selected_indices[..., None], 1),  position_embeddings[1][1].take_along_dim(selected_indices[..., None], 1))

    if data_cache['statistics']:
        assert not use_cache, "Statistics are not supported with use_cache."
        early_skipping.key_value_statistics(data_cache, key_states, value_states, query_states)

    if use_cache:
        key_states, value_states = early_skipping.update_fetch_kvcache(data_cache, key_states, value_states, query=query_states if data_cache['ESdLLM_mode']=='Query' else None, position_ids=position_ids)
        attn_mask = torch.logical_and(attention_mask.unsqueeze(1).unsqueeze(-1), data_cache['valid_index'].unsqueeze(1).unsqueeze(-2)).to(query_states.device) if attention_mask is not None else None
        early_skipping.record_sparse_kvcache(data_cache, query_states)
    else:
        data_cache['key'] = key_states.clone()
        data_cache['value'] = value_states.clone()
        data_cache['query'] = query_states.clone()
    
    if not use_cache:
        attn_mask = torch.logical_and(attention_mask.unsqueeze(1).unsqueeze(-1), data_cache['valid_index'].unsqueeze(1).unsqueeze(-2)).to(query_states.device) if attention_mask is not None else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    # is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask if isinstance(attn_mask, torch.Tensor) else None,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False, # hard coded
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, -1, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if data_cache['record_time']:
        data_cache['end_attn_event'].record()

    return attn_output, position_ids, attention_mask, position_embeddings, selected_indices