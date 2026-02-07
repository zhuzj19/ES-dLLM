# main.py
record_time = False
statistics = False
model_name = "LLaDA-Instruct"
gen_length = 256
token_per_step = 1
block_length = 64
parallel_mode = False
threshold = 0.9 # for confidence-aware parallel decoding
delay_eos_generation = True # whether to limit the generation of the <eos> token (only generation after unmasked tokens)

# Sparse-dLLM support
sparse_kv = 1 # proportion of keys/values to keep for sparse attention
delay_step = 1

# ES-dLLM configuration
use_kvcache = True
block_update_freq = 64
prompt_update_freq = block_length
# ESdLLM_mode = None
ESdLLM_mode = "HiddenState" # "Key", "Value", "Query" or "HiddenState"
importance_score_alpha = 0.5
proportion_steps = [(1, 0), (0.5, 0.125), (0.25, 0.25)]