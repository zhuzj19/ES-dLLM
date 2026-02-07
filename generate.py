import torch
import numpy as np
import torch.nn.functional as F
import time

def prepare_data_caches(kv_shape: tuple, real_position: torch.Tensor, hidden_dim: int, n_layers: int, n_heads: int, device: torch.device, ESdLLM_mode: str, importance_score_alpha: float, skip_layers: list[bool], record_time: bool, statistics: bool, use_cfg_scale=False)-> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ESdLLM_enable = ESdLLM_mode is not None
    q_shape = (kv_shape[0], n_heads, kv_shape[2], kv_shape[3])
    if use_cfg_scale:
        kv_shape = (kv_shape[0] * 2, kv_shape[1], kv_shape[2], kv_shape[3])  # Double the batch size for CFG
        q_shape = (q_shape[0] * 2, q_shape[1], q_shape[2], q_shape[3])
        real_position = real_position.repeat(2, 1)
    all_confidence = torch.zeros((kv_shape[0], kv_shape[2]), dtype=torch.bfloat16, device=device)
    confidence_coefficient = torch.ones((kv_shape[0], kv_shape[2]), dtype=torch.bfloat16, device=device) if ESdLLM_enable else None
    absdiff_coefficient = torch.ones((kv_shape[0], kv_shape[2]), dtype=torch.bfloat16, device=device) if ESdLLM_enable else None
    constant_coefficient = torch.zeros((kv_shape[0], kv_shape[2]), dtype=torch.bfloat16, device=device) if ESdLLM_enable else None
    data_caches = [{'key': torch.zeros(kv_shape, dtype=torch.bfloat16, device=device), 
                'value': torch.zeros(kv_shape, dtype=torch.bfloat16, device=device),
                'query': torch.zeros(q_shape, dtype=torch.bfloat16, device=device) if (ESdLLM_mode == 'Query' and skip_layers[i]) or statistics else None,
                'real_position': real_position,  # Real position for rotary embeddings

                'sparse_kv_ratio': None,
                'sparse_key': None,
                'sparse_value': None,
                'record_sparse_kv': False,
                'inference_start_index': None,
                'inference_end_index': None,
                'valid_index': None,

                'debug': False,
                'cfg_scale': use_cfg_scale,

                'ES_token_cnt': None, # remaining token cnt pf each layer
                'last_confidence': all_confidence if ESdLLM_enable else None,
                'ESdLLM_mode': ESdLLM_mode,
                'importance_score_alpha': importance_score_alpha,
                'confidence_coefficient': confidence_coefficient,
                'absdiff_coefficient': absdiff_coefficient,
                'constant_coefficient': constant_coefficient,
                'position_ids': None,  # Position index for ES-dLLM 

                'statistics': statistics, # whether to record statistics
                'hidden_state': torch.zeros((kv_shape[0], kv_shape[2], hidden_dim), dtype=torch.bfloat16, device=device) if statistics or (ESdLLM_mode == 'HiddenState' and skip_layers[i]) else None,
                'record_time': record_time,
                'start_proj_event': torch.cuda.Event(enable_timing=True) if record_time else None,
                'end_proj_event': torch.cuda.Event(enable_timing=True) if record_time else None,
                'start_attn_event': torch.cuda.Event(enable_timing=True) if record_time else None,
                'end_attn_event': torch.cuda.Event(enable_timing=True) if record_time else None,
                'start_ffn_event': torch.cuda.Event(enable_timing=True) if record_time else None,
                'end_ffn_event': torch.cuda.Event(enable_timing=True) if record_time else None, 
                'start_importance_score': torch.cuda.Event(enable_timing=True) if record_time else None,
                'end_importance_score': torch.cuda.Event(enable_timing=True) if record_time else None
                } for i in range(n_layers)]
    # data_caches.append({'hidden_state': None, 'hidden_state_similarity': None})  # For the final layer
    return data_caches, all_confidence, confidence_coefficient, absdiff_coefficient, constant_coefficient

@torch.no_grad()
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

@torch.no_grad()
def sample_tokens_LLaDA(logits, pos, mask_index, all_confidence, decode_token_cnt=1, eos_token = -1, threshold:float=None, temperature=0.0, remasking='low_confidence', exist_eos=None, parallel_mode=False):
    """
    pos: (batch_size, decoding_length), the position (in x) of the tokens to be sampled.
    """
    # add_gumbel_noise
    batch_size = logits.shape[0]
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    confidence_with_unmask = x0_p.clone()
    if eos_token != -1:
        x0_p[torch.logical_and(torch.logical_and(x0 == eos_token, ~exist_eos.unsqueeze(1).expand_as(x0)), mask_index.gather(-1, torch.max(pos - 1, other=torch.zeros_like(pos))))] = 0.0
    x0_p[~mask_index.gather(-1, pos)] = 0.0
    all_confidence.scatter_(1, pos, x0_p)
    if parallel_mode:
        unmask_tokens = x0_p > threshold
        unmask_tokens.scatter_(-1, torch.topk(x0_p, k=1, dim=-1).indices, True)
        unmask_tokens = torch.logical_and(unmask_tokens, x0_p > 0.0)
        # selected_index = [pos[i][unmask_tokens[i]] for i in range(pos.shape[0])]
        selected_index = [unmask_tokens[i].nonzero().squeeze(1) for i in range(pos.shape[0])]
    else:
        if threshold is not None:
            x0_p_threshold_cnt = torch.min(torch.sum(x0_p > threshold, dim=-1)).item()
            # print(torch.sum(x0_p > threshold, dim=-1))
            decode_token_cnt = min(decode_token_cnt, x0_p_threshold_cnt)
            decode_token_cnt = max(decode_token_cnt, 1)
        selected_index = torch.topk(x0_p, k=decode_token_cnt, dim=-1).indices
    return x0, confidence_with_unmask, selected_index

@torch.no_grad()
def top_p_logits(logits: torch.Tensor, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits, mask

def top_k_logits(logits: torch.Tensor, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    # p = F.softmax(logits, dim=-1)
    return logits, indices_to_remove

@torch.no_grad()
def sample_tokens_utils_Dream(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    tmp_logits = logits.clone()
    if top_p is not None and top_p < 1:
        tmp_logits, mask1 = top_p_logits(tmp_logits, top_p)
        mask = mask | mask1
    if top_k is not None:
        tmp_logits, mask1 = top_k_logits(tmp_logits, top_k)
        mask = mask | mask1
    probs = torch.softmax(logits, dim=-1)
    probs = probs.masked_fill(mask, 0.0)

    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            # confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            confidence, _ = probs.max(dim=-1) # use max prob as confidence
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

@torch.no_grad()
def sample_tokens_Dream(logits, position_ids, mask_index, all_confidence, decode_token_cnt=1, eos_token=-1, threshold=None, temperature=0.0, top_p=1.0, top_k=50, exist_eos=None, parallel_mode=False):
    # use maskgit_plus sampling
    confidence, x0 = sample_tokens_utils_Dream(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    # full_confidence = torch.full_like(x, -torch.inf, device=model.device, dtype=logits.dtype)
    confidence_with_unmask = confidence.clone()
    confidence[~mask_index.gather(-1, position_ids)] = 0.0
    all_confidence.scatter_(1, position_ids, confidence)
    if eos_token != -1:
        confidence[torch.logical_and(torch.logical_and(x0 == eos_token, ~exist_eos.unsqueeze(1).expand_as(x0)), mask_index.gather(-1, torch.max(position_ids - 1, other=torch.zeros_like(position_ids))))] = 0.0
    if parallel_mode:
        unmask_tokens = confidence > threshold
        unmask_tokens.scatter_(-1, torch.topk(confidence, k=1, dim=-1).indices, True)
        unmask_tokens = torch.logical_and(unmask_tokens, confidence > 0.0)
        # selected_index = [pos[i][unmask_tokens[i]] for i in range(pos.shape[0])]
        transfer_index = [unmask_tokens[i].nonzero().squeeze(1) for i in range(position_ids.shape[0])]
    else:
        if threshold is not None:
            confidence_threshold_cnt = torch.min(torch.sum(confidence > threshold, dim=-1)).item()
            # print(torch.sum(confidence > threshold, dim=-1))
            decode_token_cnt = min(decode_token_cnt, confidence_threshold_cnt)
            decode_token_cnt = max(decode_token_cnt, 1)
        if decode_token_cnt > 0:
            _, transfer_index = torch.topk(confidence, decode_token_cnt)
    return x0, confidence_with_unmask, transfer_index

def statistics_per_step(results: dict, data_caches: list, x0: torch.Tensor, x0_p: torch.Tensor, step: int, n_layers: int, mask_index: torch.Tensor, hidden_state_absdiff_all: torch.Tensor, key_absdiff_all: torch.Tensor, value_absdiff_all: torch.Tensor, query_absdiff_all: torch.Tensor, confidence_diff_all: torch.Tensor):
    if len(results['confidence']) > 0:
        results['confidence_diff'].append(torch.abs(x0_p - results['confidence'][-1]))
    else:
        results['confidence_diff'].append(torch.abs(x0_p))
    results['confidence'].append(x0_p)
    if step == 0:
        results['token_change_cnt'].append([])
    if len(results['predicted_token']) > 0:
        results['token_change_cnt'][-1].append(torch.sum((x0[mask_index] != results['predicted_token'][-1][mask_index])).item() / x0.shape[0])
    results['predicted_token'].append(x0)

    results['hidden_state_similarity'].append(torch.stack([data_caches[block_idx]['hidden_state_similarity'] for block_idx in range(len(data_caches))], dim=-1))
    results['hidden_state_absdiff'].append(torch.stack([data_caches[block_idx]['hidden_state_absdiff'] for block_idx in range(len(data_caches))], dim=-1))
    results['hidden_state_absdiff_layersim'].append(torch.stack([torch.cosine_similarity(results['hidden_state_absdiff'][-1].float()[:, :, i], results['hidden_state_absdiff'][-1].float()[:, :, -1]) for i in range(n_layers)]))
    results['key_similarity'].append(torch.stack([data_caches[block_idx]['key_similarity'] for block_idx in range(len(data_caches))], dim=-1))
    results['key_absdiff'].append(torch.stack([data_caches[block_idx]['key_absdiff'] for block_idx in range(len(data_caches))], dim=-1))
    results['key_absdiff_layersim'].append(torch.stack([torch.cosine_similarity(results['key_absdiff'][-1].float()[:, :, i], results['key_absdiff'][-1].float()[:, :, -1]) for i in range(n_layers)]))
    results['value_similarity'].append(torch.stack([data_caches[block_idx]['value_similarity'] for block_idx in range(len(data_caches))], dim=-1))
    results['value_absdiff'].append(torch.stack([data_caches[block_idx]['value_absdiff'] for block_idx in range(len(data_caches))], dim=-1))
    results['value_absdiff_layersim'].append(torch.stack([torch.cosine_similarity(results['value_absdiff'][-1].float()[:, :, i], results['value_absdiff'][-1].float()[:, :, -1]) for i in range(n_layers)]))
    results['query_similarity'].append(torch.stack([data_caches[block_idx]['query_similarity'] for block_idx in range(len(data_caches))], dim=-1))
    results['query_absdiff'].append(torch.stack([data_caches[block_idx]['query_absdiff'] for block_idx in range(len(data_caches))], dim=-1))
    results['query_absdiff_layersim'].append(torch.stack([torch.cosine_similarity(results['query_absdiff'][-1].float()[:, :, i], results['query_absdiff'][-1].float()[:, :, -1]) for i in range(n_layers)]))
    if step != 0: # absdiff in first step of each block is meaningless
        hidden_state_absdiff_all = torch.concat([hidden_state_absdiff_all, torch.stack([results['hidden_state_absdiff'][-1][:, :, l][mask_index] for l in range(n_layers)])], dim=-1)
        key_absdiff_all = torch.concat([key_absdiff_all, torch.stack([results['key_absdiff'][-1][:, :, l][mask_index] for l in range(n_layers)])], dim=-1)
        value_absdiff_all = torch.concat([value_absdiff_all, torch.stack([results['value_absdiff'][-1][:, :, l][mask_index] for l in range(n_layers)])], dim=-1)
        query_absdiff_all = torch.concat([query_absdiff_all, torch.stack([results['query_absdiff'][-1][:, :, l][mask_index] for l in range(n_layers)])], dim=-1)
        confidence_diff_all = torch.concat([confidence_diff_all, results['confidence_diff'][-1][mask_index]], dim=-1)
    return hidden_state_absdiff_all, key_absdiff_all, value_absdiff_all, query_absdiff_all, confidence_diff_all

def statistics_final(results: dict, n_layers: int, hidden_state_absdiff_all: torch.Tensor, key_absdiff_all: torch.Tensor, value_absdiff_all: torch.Tensor, query_absdiff_all: torch.Tensor, confidence_diff_all: torch.Tensor):
    results['hidden_state_absdiff_pass'] = torch.mean(torch.mean(torch.stack(results['hidden_state_absdiff_layersim'][1:]), dim=0), dim=1)
    results['key_absdiff_pass'] = torch.mean(torch.mean(torch.stack(results['key_absdiff_layersim'][1:]), dim=0), dim=1)
    results['value_absdiff_pass'] = torch.mean(torch.mean(torch.stack(results['value_absdiff_layersim'][1:]), dim=0), dim=1)
    results['query_absdiff_pass'] = torch.mean(torch.mean(torch.stack(results['query_absdiff_layersim'][1:]), dim=0), dim=1)
    results['hidden_state_correlation_all'] = [torch.corrcoef(torch.stack([hidden_state_absdiff_all[l][confidence_diff_all > 0], confidence_diff_all[confidence_diff_all > 0]]))[0, 1].item() for l in range(n_layers)]
    results['key_correlation_all'] = [torch.corrcoef(torch.stack([key_absdiff_all[l][confidence_diff_all > 0], confidence_diff_all[confidence_diff_all > 0]]))[0, 1].item() for l in range(n_layers)]
    results['value_correlation_all'] = [torch.corrcoef(torch.stack([value_absdiff_all[l][confidence_diff_all > 0], confidence_diff_all[confidence_diff_all > 0]]))[0, 1].item() for l in range(n_layers)]
    results['query_correlation_all'] = [torch.corrcoef(torch.stack([query_absdiff_all[l][confidence_diff_all > 0], confidence_diff_all[confidence_diff_all > 0]]))[0, 1].item() for l in range(n_layers)]

total_effective_tokens = 0
request_cnt = 0
total_importance_time_global = 0.0
total_time_global = 1e-9

def print_statistics():
    global total_effective_tokens, request_cnt
    print("Effective tokens per request: {:.2f}".format(total_effective_tokens / request_cnt))
    print("Importance score time ratio: {:.4f}% ({:2f}/{:.2f})".format(100.0 * total_importance_time_global / total_time_global, total_importance_time_global, total_time_global))

@ torch.no_grad()
def batch_generate(model, input_ids, mask, generation_kwargs: dict = {}):
    '''
    Args:
        model: Mask predictor.
        input_ids: The input tokens. A tensor of shape (batch_size, L).
        mask: The input mask. A tensor of shape (batch_size, L).

        generation_kwargs: A dictionary of generation arguments, including:
        gen_length: Generated answer length. (default: 256)
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking. (default: 256)
        token_per_step: Number of tokens to be generated per step (only valid when parallel_mode is False). (default: 1)
        parallel_mode: Whether to use confidence-aware parallel decoding. (default: False)
        threshold: Confidence threshold for parallel decoding. (default: None)
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        delay_eos_generation: Whether to limit the generation of the <eos> token (only generation after unmasked tokens). (default: False)
        use_kvcache: Whether to use kv-cache. (default: True)
        top_p: Top-p sampling parameter. (Dream specific)
        top_k: Top-k sampling parameter. (Dream specific)
        ESdLLM_mode: None for no ES-dLLM, or "Key", "Value", "Query" or "HiddenState". (default: None)
        importance_score_alpha: Alpha for importance score calculation. (default: None)
        prompt_update_freq: Prompt update frequency. (default: 256)
        block_update_freq: Block update frequency. (default: 256)
        statistics: Whether to record statistics. (default: False)
        print_log: Whether to print log. (default: False)
        record_time: Whether to record time. (default: False)
    Returns:
        output_ids: The output tokens. A tensor of shape (batch_size, L + gen_length).
    '''
    if model.config.model_type == 'llada':
        model_type = 'llada'
        mask_id = 126336
        eos_token = 126081
        n_layers = model.config.n_layers
        n_heads = model.config.n_heads
        n_kv_heads = model.config.n_kv_heads
        hidden_dim = model.config.d_model
        token_offset = 0
    elif model.config.model_type == 'Dream':
        model_type = 'Dream'
        mask_id = 151666
        eos_token = 151643
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        n_kv_heads = model.config.num_key_value_heads
        hidden_dim = model.config.hidden_size
        token_offset = 1
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}. Please use 'llada' or 'Dream'.")

    gen_length = generation_kwargs.get('gen_length', 256)
    block_length = generation_kwargs.get('block_length', 256)
    token_per_step = generation_kwargs.get('token_per_step', 1)
    parallel_mode = generation_kwargs.get('parallel_mode', False)
    sparse_kv = generation_kwargs.get('sparse_kv', False)
    delay_step = generation_kwargs.get('delay_step', -1)
    threshold = generation_kwargs.get('threshold', None) # for parallel decodin
    temperature = generation_kwargs.get('temperature', 0.0)
    cfg_scale = generation_kwargs.get('cfg_scale', 0.0)
    delay_eos_generation = generation_kwargs.get('delay_eos_generation', False)
    use_kvcache = generation_kwargs.get('use_kvcache', True)

    # Dream
    top_p = generation_kwargs.get('top_p', 0.95)
    top_k = generation_kwargs.get('top_k', 50)

    # ES-dLLM
    ESdLLM_mode = generation_kwargs.get('ESdLLM_mode', None) # None for no ES-dLLM
    importance_score_alpha = generation_kwargs.get('importance_score_alpha', None)
    prompt_update_freq = generation_kwargs.get('prompt_update_freq', 256)
    block_update_freq = generation_kwargs.get('block_update_freq', 256)

    # update_max_length = generation_kwargs.get('update_max_length', False) # deprecated
    # redecoding = generation_kwargs.get('redecoding', False) # deprecated
    # redecoding_threshold = generation_kwargs.get('redecoding_threshold', None) # deprecated

    do_statistics = generation_kwargs.get('statistics', False)
    print_log = generation_kwargs.get('print_log', False)
    record_time = generation_kwargs.get('record_time', False)

    if not delay_eos_generation:
        eos_token = -1

    assert threshold is not None or not parallel_mode, "Please provide threshold for parallel decoding."
    assert not do_statistics or (not use_kvcache and not parallel_mode), "Statistics is not compatible with kv-cache or parallel-mode."
    if ESdLLM_mode is not None:
        assert importance_score_alpha is not None, "Please provide importance_score_alpha for ES-dLLM."
        assert use_kvcache, "ES-dLLM requires kv-cache."

    batch_size, prompt_len = input_ids.shape
    max_length = prompt_len + gen_length
    x = F.pad(input_ids, (0, max_length + token_offset - input_ids.shape[1]), value=mask_id).to(model.device)
    prompt_index = (x != mask_id)
    if mask is not None and torch.any(mask == 0.0):
        mask = F.pad(mask, (0, gen_length), value=1.0)  # Pad attention mask to match the length of x
        real_position = torch.cumsum(mask, dim=-1).to(x.device) - 1  # Real position for rotary embeddings
        attention_mask = torch.logical_and(mask.unsqueeze(1).unsqueeze(-2), mask.unsqueeze(1).unsqueeze(-1)).to(model.device)   # (batch_size, 1, L + gen_length, L + gen_length)
    else:
        mask = torch.ones((batch_size, max_length), dtype=torch.bfloat16, device=x.device)
        attention_mask = torch.ones((batch_size, 1, max_length, max_length), dtype=torch.bool, device=x.device)  # Full attention mask
        real_position = torch.arange(max_length, dtype=torch.long, device=x.device).expand(batch_size, -1)
    mask = mask.to(model.device)

    decode_proportions = None
    if ESdLLM_mode is not None:
        decode_proportions = [[p for p, s in generation_kwargs.get('proportion_steps') if i >= s * n_layers][-1] for i in range(n_layers)]
        skip_layers = [decode_proportions[i] != decode_proportions[i - 1] for i in range(1, n_layers)]
        skip_layers = [decode_proportions[0] != 1] + skip_layers
        if print_log:
            print(f"FLOPs proportion: {(sum(decode_proportions) + 1 - decode_proportions[-1]) / len(decode_proportions):.2f}")
        # print(f"FLOPs proportion: {(sum(decode_proportions) + 1 - decode_proportions[-1]) / len(decode_proportions):.2f}")
    else:
        skip_layers = [False] * n_layers  # No skip layers for non-ESdLLM mode

    data_caches, all_confidence, confidence_coef, absdiff_coef, constant_coef = prepare_data_caches(
        kv_shape=(batch_size, n_kv_heads, max_length, hidden_dim // n_heads),
        real_position=real_position,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        device=model.device,
        use_cfg_scale=(cfg_scale > 0.),
        ESdLLM_mode=ESdLLM_mode,
        importance_score_alpha=importance_score_alpha,
        skip_layers=skip_layers,

        statistics=do_statistics,
        record_time=record_time
    )
    decoding_confidence = torch.zeros((batch_size, max_length), dtype=torch.bfloat16, device=model.device)
    decoding_confidence[:, :prompt_len] = 1.0  # Set confidence for prompt tokens to 1.0
    # confidence_coef[:, :] = 0.0
    # absdiff_coef[:, :] = 0.0

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps = gen_length // token_per_step
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_projection_time = 0.0
    total_attention_time = 0.0
    total_ffn_time = 0.0
    total_importance_score_time = 0.0
    all_update = True

    results = {
        'confidence': [], # seq_len
        'confidence_diff': [], # seq_len
        'key_similarity': [], # seq_len * num_blocks
        'key_absdiff': [], # seq_len * num_blocks
        'key_absdiff_layersim': [], # num_blocks * batch_size
        'value_similarity': [], # seq_len * num_blocks
        'value_absdiff': [], # seq_len * num_blocks
        'value_absdiff_layersim': [], # num_blocks * batch_size
        'query_similarity': [], # seq_len * num_blocks
        'query_absdiff': [], # seq_len * num_blocks
        'query_absdiff_layersim': [], # num_blocks * batch_size
        'hidden_state_similarity': [], # seq_len * (num_blocks + 1)
        'hidden_state_absdiff': [], # seq_len * (num_blocks + 1)
        'hidden_state_absdiff_layersim': [], # num_blocks * batch_size
        'predicted_token': [], # seq_len
        'token_change_cnt': []
    } # step

    st_time = time.time()

    hidden_state_absdiff_all = torch.zeros((n_layers, 0), device=model.device)
    key_absdiff_all = torch.zeros((n_layers, 0), device=model.device)
    value_absdiff_all = torch.zeros((n_layers, 0), device=model.device)
    query_absdiff_all = torch.zeros((n_layers, 0), device=model.device)
    confidence_diff_all = torch.zeros((0), device=model.device)

    # last_pos = None
    last_decode_cnt = 0
    last_decode_tokens = None

    step_cnt = 0

    for num_block in range(num_blocks):
        start_index = prompt_len + num_block * block_length
        end_index = prompt_len + (num_block + 1) * block_length
        start_index -= token_offset
        end_index -= token_offset
        forward_start_index = 0
        forward_end_index = max_length
        for step in range(block_length):
            if ESdLLM_mode is not None:
                if step_cnt == 0:
                    forward_start_index = 0
                    forward_end_index = max_length
                    all_update = True
                else:
                    forward_start_index = start_index if step != 0 else leftmost_decode_pos
                    if step_cnt % prompt_update_freq == 0:
                        forward_start_index = 0
                    forward_end_index = end_index
                    all_update = (step_cnt % block_update_freq == 0) or (step_cnt % prompt_update_freq == 0)
                    all_update = all_update or (step == delay_step and sparse_kv != 1.0)
                    # if update_max_length and step_cnt % prompt_update_freq == 0:
                    #     forward_end_index = max_length
            else:
                if step == 0 or not use_kvcache:
                    forward_start_index = 0
                    forward_end_index = max_length
                else:
                    forward_start_index = start_index
                    forward_end_index = end_index
                all_update = True
            # print(forward_start_index, forward_end_index, all_update)
            if not all_update:
                # ES-dLLM: token_cnt for each layer
                for block_idx in range(n_layers):
                    data_caches[block_idx]['ES_token_cnt'] = max(int(decode_proportions[block_idx] * (forward_end_index - forward_start_index)), min(last_decode_cnt + 2, end_index - start_index))
            else:
                for block_idx in range(n_layers):
                    data_caches[block_idx]['ES_token_cnt'] = None
            # Sparse Attention
            for block_idx in range(n_layers):
                data_caches[block_idx]['inference_start_index'] = start_index
                data_caches[block_idx]['inference_end_index'] = end_index
                if step == 0:
                    data_caches[block_idx]['sparse_key'] = None
                    data_caches[block_idx]['sparse_value'] = None
                    data_caches[block_idx]['valid_index'] = mask
                if sparse_kv != 1.0 and step == delay_step:
                    data_caches[block_idx]['record_sparse_kv'] = True
                    data_caches[block_idx]['sparse_kv_ratio'] = sparse_kv
                else:
                    data_caches[block_idx]['record_sparse_kv'] = False

            if token_offset != 0:
                mask_index = (x[:, :-token_offset] == mask_id)
                mask_index[:, :-token_offset] = mask_index[:, token_offset:].clone()
            else:
                mask_index = (x == mask_id)
            mask_index[:, :start_index] = False
            mask_index[:, end_index:] = False
            if mask_index.sum() == 0:
                step_cnt += block_length - step
                break

            if not all_update:
                constant_coef[:, :] = 0.0
                # consistent decoding
                # if step != 0:
                #     constant_coef.scatter_(-1, last_pos, 1.0)
                # constant_coef[~mask_index] = 0.0
                for i in range(batch_size):
                    nzero = mask_index[i].nonzero()
                    if nzero.shape[0] > 0:
                        leftmost_mask_index = mask_index[i].nonzero()[0].item()
                        constant_coef[i, leftmost_mask_index] = 5.0
                    if last_decode_tokens is not None and importance_score_alpha < 1.0:
                        constant_coef[i, last_decode_tokens[i]] = 3.0
            pos = torch.arange(forward_start_index, forward_end_index, dtype=torch.long, device=x.device).expand(batch_size, -1)
            input_x = torch.gather(x, 1, pos)
            cur_mask = torch.gather(mask, 1, pos)
            # input_mask = torch.logical_and(cur_mask.unsqueeze(1).unsqueeze(-1), mask.unsqueeze(1).unsqueeze(-2)).to(model.device) 
            # input_mask = torch.gather(attention_mask, 2, pos[:, None, :, None].expand(-1, -1, -1, attention_mask.shape[-1]))
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                un_x = un_x.gather(1, pos)
                input_x = torch.cat([input_x, un_x], dim=0)
                input_mask = input_mask.repeat(2, 1, 1, 1)
                pos = pos.repeat(2, 1)
            logits = model(input_x, attention_mask=cur_mask, position_ids=pos, data_caches=data_caches, use_cache=use_kvcache).logits
            if not all_update:
                pos = data_caches[n_layers - 1]['position_ids']
            if cfg_scale > 0.:
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                pos = pos[:batch_size, :]
            # last_pos = pos

            if forward_start_index == 0 and not do_statistics:
                # truncate the useless part to save memory
                logits = logits[:, start_index: end_index]
                pos = pos[:, start_index: end_index]

            if model_type == 'llada':
                x0, x0_p, select_index = sample_tokens_LLaDA(logits, pos, mask_index, all_confidence, decode_token_cnt=token_per_step, threshold=threshold, temperature=temperature, eos_token=eos_token, exist_eos=(x[:, prompt_len:] == eos_token).any(dim=-1), parallel_mode=parallel_mode)
            elif model_type == 'Dream':
                x0, x0_p, select_index = sample_tokens_Dream(logits, pos, mask_index, all_confidence, decode_token_cnt=token_per_step, threshold=threshold, temperature=temperature, top_p=top_p, top_k=top_k, eos_token=eos_token, exist_eos=(x[:, prompt_len:] == eos_token).any(dim=-1), parallel_mode=parallel_mode)
            if parallel_mode:
                last_decode_cnt = max([idx.shape[0] for idx in select_index])
                for i in range(batch_size):
                    x[i, pos[i][select_index[i]] + token_offset] = x0[i, select_index[i]]
                    decoding_confidence[i, pos[i][select_index[i]]] = x0_p[i, select_index[i]]
            else:
                last_decode_cnt = select_index.shape[-1]
                x.scatter_(1, pos.gather(-1, select_index) + token_offset, x0.gather(-1, select_index))
                decoding_confidence.scatter_(1, pos.gather(-1, select_index), x0_p.gather(-1, select_index))
            if do_statistics:
                hidden_state_absdiff_all, key_absdiff_all, value_absdiff_all, query_absdiff_all, confidence_diff_all = statistics_per_step(results, data_caches, x0, x0_p, step, n_layers, mask_index, hidden_state_absdiff_all, key_absdiff_all, value_absdiff_all, query_absdiff_all, confidence_diff_all)

            if parallel_mode:
                last_decode_tokens = [pos[i, select_index[i]] for i in range(batch_size)]
                for i in range(batch_size):
                    all_confidence[i, last_decode_tokens[i]] = 0.0       
                leftmost_decode_pos = min([min(tokens, default=10000) for tokens in last_decode_tokens]).item()
                print_token = [x0[i, select_index[i]].tolist() for i in range(batch_size)]
                print_confidence = [x0_p[i, select_index[i]].tolist() for i in range(batch_size)]
                last_decode_tokens = [tokens.tolist() for tokens in last_decode_tokens]
            else:
                last_decode_tokens = pos.gather(-1, select_index)
                all_confidence.scatter_(-1, last_decode_tokens, 0.0)
                leftmost_decode_pos = last_decode_tokens.min().item()
                last_decode_tokens = last_decode_tokens.tolist()
                print_token = x0.gather(-1, select_index).tolist()
                print_confidence = x0_p.gather(-1, select_index).tolist()
            if print_log:
                print(f"Step {step + 1}/{steps_per_block}, Block {num_block + 1}/{num_blocks}, Select index: {last_decode_tokens}, token: {print_token}, confidence: {print_confidence}") # Select index + 1 for Dream
            
            if record_time:
                torch.cuda.synchronize()
                total_projection_time += np.sum([cache['start_proj_event'].elapsed_time(cache['end_proj_event']) / 1000.0 for cache in data_caches if 'start_proj_event' in cache])
                total_attention_time += np.sum([cache['start_attn_event'].elapsed_time(cache['end_attn_event']) / 1000.0 for cache in data_caches if 'start_attn_event' in cache])
                total_ffn_time += np.sum([cache['start_ffn_event'].elapsed_time(cache['end_ffn_event']) / 1000.0 for cache in data_caches if 'start_ffn_event' in cache])
                total_importance_score_time += np.sum([cache['start_importance_score'].elapsed_time(cache['end_importance_score']) / 1000.0 for cache in data_caches if 'start_importance_score' in cache])
            
            step_cnt += 1
    
    if do_statistics:
        statistics_final(results, n_layers, hidden_state_absdiff_all, key_absdiff_all, value_absdiff_all, query_absdiff_all, confidence_diff_all)

    torch.cuda.synchronize()
    ed_time = time.time()
    time_info = f"Total time: {ed_time - st_time:.2f} seconds\n"
    if record_time:
        time_info += f"\tTotal projection time: {total_projection_time:.4f} seconds\n"
        time_info += f"\tTotal attention time: {total_attention_time:.4f} seconds\n"
        time_info += f"\tTotal ffn time: {total_ffn_time:.4f} seconds\n"
        time_info += f"\tTotal importance score time: {total_importance_score_time:.4f} seconds"
        global total_time_global, total_importance_time_global
        total_importance_time_global += total_importance_score_time
        total_time_global += ed_time - st_time
    results['time_info'] = time_info

    if token_offset != 0:
        x = x[:, :-token_offset]
    
    global request_cnt, total_effective_tokens
    request_cnt += batch_size
    total_effective_tokens += torch.sum((x[:, prompt_len:] != eos_token)).item()
    
    return x, results