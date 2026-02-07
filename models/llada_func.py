# Modified from LLaDA Huggingface library

import torch
from typing import Optional, Tuple, List, Union
from .llada_utils_origin import LLaDAOutput, ActivationCheckpointingStrategy
import math
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import early_skipping

# RotaryEmbedding
def RotaryEmbedding_forward(self, q: torch.Tensor, k: torch.Tensor, q_pos: torch.LongTensor = None, k_pos: torch.LongTensor = None, total_len=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.config.rope_full_precision:
        q_, k_ = q.float(), k.float()
    else:
        q_, k_ = q, k

    with torch.autocast(q.device.type, enabled=False):
        query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
        pos_sin, pos_cos = self.get_rotary_embedding(total_len, q_.device)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        if q_pos is not None:
            q_ = self.apply_rotary_pos_emb(
                pos_sin.take_along_dim(q_pos[:, None, :, None], 2),
                pos_cos.take_along_dim(q_pos[:, None, :, None], 2),
                q_,
            )
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        if k_pos is not None:
            k_ = self.apply_rotary_pos_emb(
                pos_sin.take_along_dim(k_pos[:, None, :, None], 2),
                pos_cos.take_along_dim(k_pos[:, None, :, None], 2),
                k_,
            )
        else:
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
    return q_.type_as(q), k_.type_as(k)


# LLaDABlock
def LLaDABlock_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    data_cache: dict[str, torch.Tensor] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    B, T, C = q.size()  # batch size, sequence length, d_model
    dtype = k.dtype

    # Optionally apply layer norm to keys and queries.
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    # Move head forward to be next to the batch dim.
    # shape: (B, nh, T, hs)
    q = q.view(B, -1, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    k = k.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    v = v.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    if self.config.rope:
        # Apply rotary embeddings.
        q_pos, k_pos = early_skipping.get_qk_position(data_cache, query_len=q.shape[-2], key_len=k.shape[-2], position_ids=position_ids)
        q, k = self.rotary_emb(q, k, q_pos, k_pos, data_cache['real_position'].shape[-1])

    selected_indices = None
    if data_cache['ESdLLM_mode'] in ["Query", "Key", "Value"]:
        if data_cache['record_time']:
            data_cache['start_importance_score'].record()
        q, k, v, position_ids, attention_bias, selected_indices = early_skipping.skip_in_qkv(data_cache, q, k, v, position_ids, attention_bias, importance_score_alpha=data_cache['importance_score_alpha'], absdiff_cfg=data_cache['ESdLLM_mode'])
        if data_cache['record_time']:
            data_cache['end_importance_score'].record()

    if data_cache['statistics']:
        assert not use_cache, "Statistics are not supported with use_cache."
        early_skipping.key_value_statistics(data_cache, k, v, q)

    if use_cache:
        k, v = early_skipping.update_fetch_kvcache(data_cache, k, v, query=q if data_cache['ESdLLM_mode']=='Query' else None, position_ids=position_ids)
        attn_mask = torch.logical_and(attention_bias.unsqueeze(1).unsqueeze(-1), data_cache['valid_index'].unsqueeze(1).unsqueeze(-2)).to(q.device) if attention_bias is not None else None
        early_skipping.record_sparse_kvcache(data_cache, q)
    elif data_cache['statistics']:
        # Store the keys and values in the data cache for statistics.
        data_cache['key'] = k.clone()
        data_cache['value'] = v.clone()
        data_cache['query'] = q.clone()

    if not use_cache:
        attn_mask = torch.logical_and(attention_bias.unsqueeze(1).unsqueeze(-1), data_cache['valid_index'].unsqueeze(1).unsqueeze(-2)).to(q.device) if attention_bias is not None else None

    # print(q.shape, k.shape, v.shape, attn_mask.shape)
    # Get the attention scores.
    # print(q.shape, k.shape, v.shape)
    # shape: (B, nh, T, hs)
    att = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=(q.size(1) != k.size(1))
    )

    # Re-assemble all head outputs side-by-side.
    att = att.transpose(1, 2).contiguous().view(B, -1, C)

    # Apply output projection.
    return self.attn_out(att), position_ids, attention_bias, selected_indices

# LLaDALlamaBlock
def LLaDALlamaBlock_forward(
    self,
    x: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,  # new add
    attention_bias: Optional[torch.Tensor] = None,
    data_cache: dict[str, torch.Tensor] = None,
    use_cache: bool = False,
) -> torch.Tensor:
    # Get query, key, value projections.
    # shape:
    #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
    #  - for multi-query attn q: (batch_size, seq_len, d_model)
    #                      k, v: (batch_size, seq_len, d_model // n_heads)
    #  - for group query attn q: (batch_size, seq_len, d_model)
    #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

    if data_cache['record_time']:
        data_cache['start_proj_event'].record()
    x_normed = self.attn_norm(x)
    q = self.q_proj(x_normed)
    k = self.k_proj(x_normed)
    v = self.v_proj(x_normed)
    if data_cache['record_time']:
        data_cache['end_proj_event'].record()

    # Get attention scores.
    if data_cache['record_time']:
        data_cache['start_attn_event'].record()
    if self._activation_checkpoint_fn is not None:
        att, position_ids, attention_bias, selected_indices = self._activation_checkpoint_fn(  # type: ignore
            self.attention, q, k, v, position_ids, attention_bias, data_cache=data_cache, use_cache=use_cache
        )
    else:
        att, position_ids, attention_bias, selected_indices = self.attention(q, k, v, position_ids, attention_bias, data_cache=data_cache, use_cache=use_cache)

    if selected_indices is not None:
        x = x.take_along_dim(selected_indices[..., None], 1)

    # Add attention scores.
    # shape: (B, T, C)
    x = x + self.dropout(att)
    if data_cache['record_time']:
        data_cache['end_attn_event'].record()

    # Add feed-forward projection.
    # shape: (batch_size, seq_len, d_model)
    if data_cache['record_time']:
        data_cache['start_ffn_event'].record()
    og_x = x
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.ff_norm, x)
    else:
        x = self.ff_norm(x)
    x, x_up = self.ff_proj(x), self.up_proj(x)
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.act, x)
    else:
        x = self.act(x)
    x = x * x_up
    x = self.ff_out(x)
    x = self.dropout(x)
    x = og_x + x
    if data_cache['record_time']:
        data_cache['end_ffn_event'].record()
    
    if data_cache['ESdLLM_mode'] == "HiddenState":
        if data_cache['record_time']:
            data_cache['start_importance_score'].record()
        x, position_ids, attention_bias, _ = early_skipping.skip_in_hidden_state(data_cache, x, position_ids, attention_bias, importance_score_alpha=data_cache['importance_score_alpha'])
        if data_cache['record_time']:
            data_cache['end_importance_score'].record()
        early_skipping.update_hidden_state(data_cache, x, position_ids)
    elif data_cache['statistics']:
        early_skipping.update_hidden_state(data_cache, x, position_ids)

    return x, position_ids, attention_bias

# LLaDAModel
def LLaDAModel_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None, # new add
    input_embeddings: Optional[torch.FloatTensor] = None,
    attention_bias: Optional[torch.Tensor] = None, # not used
    data_caches: Optional[List[dict[str, torch.Tensor]]] = None,
    use_cache: bool = False,
    last_logits_only: bool = False,
) -> LLaDAOutput:
    # Add Basic MDM Model config check
    assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
    assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
    assert attention_bias is None, "Attention bias is not supported."

    if data_caches:
        assert len(data_caches) >= self.config.n_layers # last layer's hidden state

    batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
    if data_caches is None:
        past_length = 0
    else:
        past_length = data_caches[0]['key'].size(-2)

    # Get embeddings of input.
    # shape: (batch_size, seq_len, d_model)
    x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

    if self.config.input_emb_norm:
        x = x * (self.config.d_model**0.5)

    if not (self.config.alibi or self.config.rope):
        # Get positional embeddings.
        # shape: (1, seq_len)
        assert False, "Positional embeddings are not supported."
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        # shape: (1, seq_len, d_model)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = pos_emb + x

    # Add input + positional embeddings and apply dropout.
    # shape: (batch_size, seq_len, d_model)
    x = self.transformer.emb_drop(x)  # type: ignore

    # Apply blocks one-by-one.
    if self.config.block_group_size == 1:
        for block_idx, block in enumerate(self.transformer.blocks):
            layer_past = None if data_caches is None else data_caches[block_idx]
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                assert False, "Activation checkpointing is not supported."
                # shape: (batch_size, seq_len, d_model)
                x, position_ids, attention_mask = self._activation_checkpoint_fn(
                    block, x, position_ids=position_ids, attention_bias=attention_mask, data_cache=layer_past, use_cache=use_cache
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, position_ids, attention_mask = block(x, position_ids=position_ids, attention_bias=attention_mask, data_cache=layer_past, use_cache=use_cache)
    else:
        for group_idx, block_group in enumerate(self.transformer.block_groups):
            if data_caches[group_idx * self.config.block_group_size]['statistics']:
                assert False, "Hidden state statistics are not supported with block groups."

            layers_past = (
                None
                if data_caches is None
                else data_caches[
                    group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                ]
            )
            x = block_group(
                x, pos=position_ids, attention_bias=attention_mask, kv_cache=layers_past, use_cache=use_cache
            )
            # if attn_key_values is not None:
            #     assert cache is not None
            #     attn_key_values.extend(cache)

    if last_logits_only:
        # shape: (batch_size, 1, d_model)
        x = x[:, -1, :].unsqueeze(1)

    # Apply final layer norm.
    # shape: (batch_size, seq_len or 1, d_model)
    x = self.transformer.ln_f(x)  # type: ignore

    # Get logits.
    # shape: (batch_size, seq_len or 1, vocab_size)
    if self.config.weight_tying:
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
    else:
        logits = self.transformer.ff_out(x)  # type: ignore
    if self.config.scale_logits:
        logits.mul_(1 / math.sqrt(self.config.d_model))
    
    # print(f"Logits shape: {logits.shape}")  # Debugging line to check logits shape

    return LLaDAOutput(logits=logits, data_caches=data_caches)  # type: ignore[arg-type]

# LLaDAModelLM
def LLaDAModelLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,  # new add
    attention_bias: Optional[torch.Tensor] = None, # not used
    inputs_embeds: Optional[torch.FloatTensor] = None,
    data_caches: Optional[List[dict[str, torch.Tensor]]] = None,
    labels: Optional[torch.LongTensor] = None, # not used
    use_cache: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[Cache] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
) -> Union[Tuple, CausalLMOutputWithPast]:
    assert attention_bias is None, "Attention bias is not supported."
    if use_cache is None:
        use_cache = self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.forward(
        input_ids=input_ids,
        position_ids=position_ids,  # new add
        input_embeddings=inputs_embeds,
        attention_mask=attention_mask,
        attention_bias=attention_bias,
        data_caches=data_caches,
        use_cache=use_cache,
    )

    logits = outputs.logits
    # hidden_states = outputs.hidden_states
    key_values = [(outputs.data_caches[i]['key'], outputs.data_caches[i]['value']) for i in range(len(outputs.data_caches))] if outputs.data_caches else None
    hidden_states = [outputs.data_caches[i]['hidden_state'] for i in range(len(outputs.data_caches))] if outputs.data_caches else None

    loss = None
    if labels is not None:
        import warnings
        warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # print(f"LLaDAModelLM_forward: logits shape={logits.shape}")  # Debugging line

    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=key_values,
        hidden_states=hidden_states,
    )