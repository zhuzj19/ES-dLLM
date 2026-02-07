import torch
import torch.nn.functional as F
import numpy as np

def key_value_statistics(data_cache, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor = None):
    eps = 1e-6
    data_cache['key_similarity'] = F.cosine_similarity(data_cache['key'].transpose(1, 2).flatten(2).float(), key.transpose(1, 2).flatten(2).float(), dim=-1)
    data_cache['value_similarity'] = F.cosine_similarity(data_cache['value'].transpose(1, 2).flatten(2).float(), value.transpose(1, 2).flatten(2).float(), dim=-1)
    key_norm = (data_cache['key'].transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(key.shape[1] * key.shape[3])) + eps
    value_norm = (data_cache['value'].transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(value.shape[1] * value.shape[3])) + eps
    data_cache['key_absdiff'] = (data_cache['key'] - key).transpose(1, 2).flatten(2).abs().mean(dim=-1) / key_norm
    data_cache['value_absdiff'] = (data_cache['value'] - value).transpose(1, 2).flatten(2).abs().mean(dim=-1) / value_norm
    data_cache['key_absdiff'] = data_cache['key_absdiff'] / torch.max(data_cache['key_absdiff'].max(dim=-1, keepdim=True)[0], torch.ones((data_cache['key_absdiff'].shape[0], 1), device=data_cache['key_absdiff'].device))
    data_cache['value_absdiff'] = data_cache['value_absdiff'] / torch.max(data_cache['value_absdiff'].max(dim=-1, keepdim=True)[0], torch.ones((data_cache['value_absdiff'].shape[0], 1), device=data_cache['value_absdiff'].device))
    if query is not None:
        data_cache['query_similarity'] = F.cosine_similarity(data_cache['query'].transpose(1, 2).flatten(2).float(), query.transpose(1, 2).flatten(2).float(), dim=-1)
        query_norm = (data_cache['query'].transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(query.shape[1] * query.shape[3])) + eps
        data_cache['query_absdiff'] = (data_cache['query'] - query).transpose(1, 2).flatten(2).abs().mean(dim=-1) / query_norm
        data_cache['query_absdiff'] = data_cache['query_absdiff'] / torch.max(data_cache['query_absdiff'].max(dim=-1, keepdim=True)[0], torch.ones((data_cache['query_absdiff'].shape[0], 1), device=data_cache['query_absdiff'].device))

def update_fetch_kvcache(data_cache, key, value, query=None, position_ids=None):
    # TODO(new): support sparse kv
    assert data_cache is not None and 'key' in data_cache and 'value' in data_cache, "Data cache must contain 'key' and 'value'."
    past_query = data_cache.get('query', None)
    if position_ids is not None:
        data_cache['key'].scatter_(2, position_ids[:, None, :, None].expand_as(key), key)
        data_cache['value'].scatter_(2, position_ids[:, None, :, None].expand_as(value), value)
        if query is not None and past_query is not None:
            past_query.scatter_(2, position_ids[:, None, :, None].expand_as(query), query)
    else:
        data_cache['key'][:, :, -key.shape[-2]:, :] = key
        data_cache['value'][:, :, -value.shape[-2]:, :] = value
    if data_cache['record_time']:
        data_cache['start_importance_score'].record()
    if data_cache['sparse_key'] is not None and data_cache['sparse_value'] is not None and data_cache['inference_start_index'] != 0:
        data_cache['sparse_key'][:, :, data_cache['inference_start_index'] - data_cache['inference_end_index']:, :] = data_cache['key'][:, :, data_cache['inference_start_index']:data_cache['inference_end_index'], :]
        data_cache['sparse_value'][:, :, data_cache['inference_start_index'] - data_cache['inference_end_index']:, :] = data_cache['value'][:, :, data_cache['inference_start_index']:data_cache['inference_end_index'], :]
        past_key, past_value = data_cache['sparse_key'], data_cache['sparse_value']
        # past_key, past_value = torch.concat([data_cache['sparse_key'], data_cache['key'][:, :, data_cache['inference_start_index']:data_cache['inference_end_index'], :]], dim=2), torch.concat([data_cache['sparse_value'], data_cache['value'][:, :, data_cache['inference_start_index']:data_cache['inference_end_index'], :]], dim=2) # note: concat in original implementation is too slow
    else:
        past_key, past_value = data_cache['key'], data_cache['value']
    if data_cache['record_time']:
        data_cache['end_importance_score'].record()
    return past_key, past_value

def record_sparse_kvcache(data_cache, query: torch.Tensor):
    if data_cache['record_sparse_kv']:
        key_outside_block = torch.concat([data_cache['key'][:, :, :data_cache['inference_start_index'], :], data_cache['key'][:, :, data_cache['inference_end_index']:, :]], dim=2)
        value_outside_block = torch.concat([data_cache['value'][:, :, :data_cache['inference_start_index'], :], data_cache['value'][:, :, data_cache['inference_end_index']:, :]], dim=2)

        # from Sparse-dLLM
        # q_block: [B, n_heads, block_len, head_dim]
        # cached_k: [B, n_kv_heads, seq_len, head_dim]
        if key_outside_block.size(1) != query.size(1): # GQA
            filtered_cached_k_attn = key_outside_block.repeat_interleave(
                query.size(1) // key_outside_block.size(1), dim=1
            )
        else:
            filtered_cached_k_attn = key_outside_block
        avg_q = query.mean(dim=-2)
        scores = torch.matmul(avg_q.unsqueeze(-2), filtered_cached_k_attn.transpose(-2, -1)).squeeze(-2)
        importance = scores.mean(dim=1)
        kernel_size = 3
        importance = F.max_pool1d(
            importance.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze(1)
        keep_num = int(importance.size(-1) * data_cache['sparse_kv_ratio'])
        _, keep_indices = torch.topk(importance, k=keep_num, dim=-1)
        keep_indices = keep_indices.squeeze(0)
        # print(keep_indices.shape)

        new_valid_index = torch.concat([data_cache['valid_index'][:, :data_cache['inference_start_index']], data_cache['valid_index'][:, data_cache['inference_end_index']:]], dim=1).take_along_dim(keep_indices, 1)
        new_valid_index = torch.concat([new_valid_index, data_cache['valid_index'][:, data_cache['inference_start_index']: data_cache['inference_end_index']]], dim=1)
        data_cache['valid_index'] = new_valid_index
        data_cache['sparse_key'] = key_outside_block.take_along_dim(keep_indices.unsqueeze(1).unsqueeze(3), 2)
        data_cache['sparse_key'] = F.pad(data_cache['sparse_key'], (0, 0, 0, data_cache['inference_end_index'] - data_cache['inference_start_index']), value=0.0)
        data_cache['sparse_value'] = value_outside_block.take_along_dim(keep_indices.unsqueeze(1).unsqueeze(3), 2)
        data_cache['sparse_value'] = F.pad(data_cache['sparse_value'], (0, 0, 0, data_cache['inference_end_index'] - data_cache['inference_start_index']), value=0.0)

def get_importance_score(confidence, absdiff, alpha = None):
    importance_score = confidence * alpha + absdiff * (1 - alpha)
    if alpha > 0.0:
        _, top2 = confidence.topk(2, dim=-1, largest=True, sorted=False)
        special_score = torch.zeros_like(importance_score)
        special_score.scatter_(1, top2, 2.0) # ensure existing masked tokens
        importance_score += special_score
    return importance_score

def get_qk_position(data_cache, query_len = None, key_len = None, position_ids=None):
    q_pos, k_pos = None, None
    if data_cache is not None and "real_position" in data_cache:
        if position_ids is not None:
            q_pos = data_cache["real_position"].take_along_dim(position_ids, 1)
            if query_len is not None:
                assert q_pos.shape[1] == query_len, f"Expected query position length {query_len}, but got {q_pos.shape[1]}."
        else:
            q_pos = data_cache["real_position"][:, :query_len]
        k_pos = q_pos
        # if key_len is not None:
        #     if key_len == query_len:
        #         k_pos = q_pos
        #     else:
        #         k_pos = data_cache["real_position"][:, :key_len]
        # else:
        #     k_pos = data_cache["real_position"]
    elif position_ids is not None:
        q_pos = position_ids
    return q_pos, k_pos

def skip_in_qkv(data_cache, query, key, value, position_ids, attention_mask, importance_score_alpha=None, absdiff_cfg="Key"):
    eps = 1e-6
    selected_indices = None
    if data_cache['ES_token_cnt'] is not None:
        assert position_ids is not None, "Position indices must be provided when ES_token_cnt is used."
        if data_cache['ES_token_cnt'] != position_ids.shape[1]:
            if absdiff_cfg == "Query":
                last_query = data_cache['query'].take_along_dim(position_ids.unsqueeze(1).unsqueeze(3), 2) if data_cache['query'] is not None else torch.zeros(query.shape, device=query.device)
                query_norm = (last_query.transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(query.shape[1] * query.shape[3])) + eps
                absdiff_q = (query - last_query).abs().transpose(1, 2).flatten(2).mean(dim=-1) / query_norm
                absdiff = absdiff_q
            elif absdiff_cfg == "Key":
                last_key = data_cache['key'].take_along_dim(position_ids.unsqueeze(1).unsqueeze(3), 2) if data_cache['key'] is not None else torch.zeros(key.shape, device=key.device)
                key_norm = (last_key.transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(key.shape[1] * key.shape[3])) + eps
                absdiff_k = (key - last_key).abs().transpose(1, 2).flatten(2).mean(dim=-1) / key_norm
                absdiff = absdiff_k
            elif absdiff_cfg == "Value":
                last_value = data_cache['value'].take_along_dim(position_ids.unsqueeze(1).unsqueeze(3), 2) if data_cache['value'] is not None else torch.zeros(value.shape, device=value.device)
                value_norm = (last_value.transpose(1, 2).flatten(2).norm(p=2, dim=-1) / np.sqrt(value.shape[1] * value.shape[3])) + eps
                absdiff_v = (value - last_value).abs().transpose(1, 2).flatten(2).mean(dim=-1) / value_norm
                absdiff = absdiff_v
            absdiff = absdiff / torch.max(absdiff.max(dim=-1, keepdim=True)[0], torch.ones((absdiff.shape[0], 1), device=absdiff.device))
            confidence = data_cache['last_confidence'].take_along_dim(position_ids, 1)
            if importance_score_alpha is None:
                confidence = confidence * data_cache['confidence_coefficient'].take_along_dim(position_ids, 1)
                absdiff = absdiff * data_cache['absdiff_coefficient'].take_along_dim(position_ids, 1)
                importance_score = confidence + absdiff
            else:
                importance_score = get_importance_score(confidence, absdiff, importance_score_alpha)
            importance_score = importance_score + data_cache['constant_coefficient'].take_along_dim(position_ids, 1)
            _, selected_indices = torch.topk(importance_score, data_cache['ES_token_cnt'], dim=-1, largest=True, sorted=False)
            if data_cache['cfg_scale']:
                selected_indices = selected_indices[:selected_indices.shape[0]//2, :].repeat(2, 1)
            query = query.take_along_dim(selected_indices[:, None, :, None], 2)
            key = key.take_along_dim(selected_indices[:, None, :, None], 2)
            value = value.take_along_dim(selected_indices[:, None, :, None], 2)
            position_ids = position_ids.take_along_dim(selected_indices, 1)
            attention_mask = attention_mask.take_along_dim(selected_indices, 1)
            # attention_mask = attention_mask.take_along_dim(selected_indices[:, None, :, None], 2)
        data_cache['position_ids'] = position_ids
    return query, key, value, position_ids, attention_mask, selected_indices

def skip_in_hidden_state(data_cache, hidden_state, position_ids, attention_mask, importance_score_alpha):
    selected_indices = None
    if data_cache['ES_token_cnt'] is not None:
        assert position_ids is not None, "Position indices must be provided when ES_token_cnt is used."
        if data_cache['ES_token_cnt'] != position_ids.shape[1]:
            last_hidden_state = data_cache['hidden_state'].take_along_dim(position_ids.unsqueeze(2), 1) if data_cache['hidden_state'] is not None else torch.zeros((hidden_state.shape[0], hidden_state.shape[1], hidden_state.shape[2]), device=hidden_state.device)
            hidden_state_norm = (last_hidden_state.norm(p=2, dim=-1) / np.sqrt(hidden_state.shape[-1])) + 1e-6
            absdiff = (hidden_state - last_hidden_state).abs().mean(dim=-1) / hidden_state_norm
            absdiff = absdiff / torch.max(absdiff.max(dim=-1, keepdim=True)[0], torch.ones((absdiff.shape[0], 1), device=absdiff.device))
            confidence = data_cache['last_confidence'].take_along_dim(position_ids, 1)
            if importance_score_alpha == "coef":
                confidence = confidence * data_cache['confidence_coefficient'].take_along_dim(position_ids, 1)
                absdiff = absdiff * data_cache['absdiff_coefficient'].take_along_dim(position_ids, 1)
                importance_score = confidence + absdiff
            else:
                importance_score = get_importance_score(confidence, absdiff, importance_score_alpha)
            importance_score = importance_score + data_cache['constant_coefficient'].take_along_dim(position_ids, 1)
            _, selected_indices = torch.topk(importance_score, data_cache['ES_token_cnt'], dim=-1, largest=True, sorted=False)
            if data_cache['cfg_scale']:
                selected_indices = selected_indices[:selected_indices.shape[0]//2, :].repeat(2, 1)
            hidden_state = hidden_state.take_along_dim(selected_indices[..., None], 1)
            position_ids = position_ids.take_along_dim(selected_indices, 1)
            # attention_mask = attention_mask.take_along_dim(selected_indices[:, None, :, None], 2)
            attention_mask = attention_mask.take_along_dim(selected_indices, 1)
        data_cache['position_ids'] = position_ids
    return hidden_state, position_ids, attention_mask, selected_indices

def update_hidden_state(data_cache, hidden_state, position_ids=None):
    if data_cache['statistics']:
        if data_cache['hidden_state'] is not None:
            data_cache['hidden_state_similarity'] = F.cosine_similarity(data_cache['hidden_state'].float(), hidden_state.float(), dim=-1)
            hidden_state_norm = (data_cache['hidden_state'].norm(p=2, dim=-1) / np.sqrt(hidden_state.shape[-1])) + 1e-6
            data_cache['hidden_state_absdiff'] = (data_cache['hidden_state'] - hidden_state).abs().mean(dim=-1) / hidden_state_norm
        else:
            data_cache['hidden_state_similarity'] = torch.zeros((hidden_state.shape[0], hidden_state.shape[1]), device=hidden_state.device)
            data_cache['hidden_state_absdiff'] = hidden_state.abs().mean(dim=-1)
        data_cache['hidden_state_absdiff'] = data_cache['hidden_state_absdiff'] / torch.max(data_cache['hidden_state_absdiff'].max(dim=-1, keepdim=True)[0], torch.ones((data_cache['hidden_state_absdiff'].shape[0], 1), device=data_cache['hidden_state_absdiff'].device))
    
    if data_cache['hidden_state'] is not None:
        if position_ids is not None:
            data_cache['hidden_state'].scatter_(1, position_ids[..., None].expand_as(hidden_state), hidden_state)
        else:
            data_cache['hidden_state'] = hidden_state.clone()