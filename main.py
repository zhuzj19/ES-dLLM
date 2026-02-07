import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import config

from models.hook_model import transform_llada_model, transform_dream_model
from generate import batch_generate

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(123)

    device = 'cuda'

    if config.model_name == "LLaDA-Instruct":
        model_path = "GSAI-ML/LLaDA-8B-Instruct"
        model_type = "LLaDA"
    elif config.model_name == "LLaDA-Base":
        model_path = "GSAI-ML/LLaDA-8B-Base"
        model_type = "LLaDA"
    elif config.model_name == "Dream-Base":
        model_path = "Dream-org/Dream-v0-Base-7B"
        model_type = "Dream"
    elif config.model_name == "Dream-Instruct":
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        model_type = "Dream"
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    if model_type == "LLaDA":
        transform_llada_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == "Dream":
        transform_dream_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left') # set left padding
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please choose 'LLaDA' or 'Dream'.")

    # The following are examples for Instruct model testing.
    messages = [
        # Answer: 18
        [ 
            {"role": "user", "content": "Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?"},
            {"role": "assistant", "content": "Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12"},
            {"role": "user", "content":"Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
        ],
        # Answer: (A)
        [
            {"role": "system", "content": "Clarify the meaning of sentences with ambiguous pronouns."},
            {"role": "user", "content": "Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.\nOptions:\n(A) The chief took the day off\n(B) The counselor took the day off\n(C) Ambiguous\nA: Let's think step by step."},
            {"role": "assistant", "content": "Let's think step by step.\nHere we need to determine who the pronoun \"they\" might be referring to. There are two possible referents for \"they\", namely the chief and the counselor. The verb \"told\" might be able to help us determine which one is more likely (if either). Let X be the chief and Y the counselor. The sentence is then of the form \"X told Y that (X or Y) did something.\"\nLet's consider Y first: \"X told Y that Y did something.\" This case does not make much sense, as Y would already have the information that Y did something, because it is information about themself.\nNow, consider X: \"X told Y that X did something.\" This makes sense, because X would be sharing some information about themself that Y might not have known before.\nBecause in this context, X is the chief and Y is the counselor, the answer should be the chief. So the answer is (A)."},
            {"role": "user", "content": "Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The patient was referred to the specialist because he had a rare skin condition.\nOptions:\n(A) The patient had a skin condition\n(B) The specialist had a skin condition\n(C) Ambiguous\nA: Let's think step by step."}
        ],
        [
            {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n"},
            {"role": "assistant", "content": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) "},
            {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n"},
            {"role": "assistant", "content": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result"},
            {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n"},
            {"role": "assistant", "content": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums"},
            {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a function to sort a given matrix in ascending order according to the sum of its rows. Your code should pass these tests:\n\nassert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]\nassert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]\nassert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]\n"}
        ]
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', padding=True, return_dict=True)

    generation_kwargs = {
        "gen_length": config.gen_length,
        "block_length": config.block_length,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "use_kvcache": config.use_kvcache,
        "token_per_step": config.token_per_step,
        "threshold": config.threshold,
        "parallel_mode": config.parallel_mode,
        "sparse_kv": config.sparse_kv,
        "delay_step": config.delay_step,
        "print_log": True,
        "record_time": config.record_time,
        "statistics": config.statistics,
        "delay_eos_generation": config.delay_eos_generation,
        # Dream specific parameters
        "top_p": 0.95,
        "top_k": 50,
        # ES-dLLM configurations
        "block_update_freq": config.block_update_freq,
        "prompt_update_freq": config.prompt_update_freq,
        "ESdLLM_mode": config.ESdLLM_mode,
        "importance_score_alpha": config.importance_score_alpha,
        "proportion_steps": config.proportion_steps,
        # deprecated
        # "update_max_length": config.update_max_length,
        # "redecoding": config.redecoding,
        # "redecoding_threshold": config.redecoding_threshold,
    }

    out, statistics = batch_generate(model, inputs.input_ids, inputs.attention_mask, generation_kwargs)
    outputs = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    for output in outputs:
        print(output)
    # print(statistics['time_info'])
    if config.statistics:
        print("Hidden state absolute pass:", statistics['hidden_state_absdiff_pass'])
        print("Key absolute pass:", statistics['key_absdiff_pass'])
        print("Value absolute pass:", statistics['value_absdiff_pass'])
        print("Query absolute pass:", statistics['query_absdiff_pass'])
        print("Hidden state correlation with confidence per layer:", statistics['hidden_state_correlation_all'])
        print("Key correlation with confidence per layer:", statistics['key_correlation_all'])
        print("Value correlation with confidence per layer:", statistics['value_correlation_all'])
        print("Query correlation with confidence per layer:", statistics['query_correlation_all'])
        print("token_change_cnt", statistics['token_change_cnt'])

if __name__ == '__main__':
    main()
