'''
This file is inspired by the code from https://github.com/ML-GSAI/LLaDA
'''
import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate, setup_parser
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate import batch_generate, print_statistics
from models.hook_model import transform_llada_model, transform_dream_model
import time
import os

import argparse


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()
        if self.model.config.model_type == 'llada':
            transform_llada_model(self.model)
        elif self.model.config.model_type == 'Dream':
            transform_dream_model(self.model)

        if self.accelerator is not None:
            # self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            if device:
                self.device = torch.device(device)
            else:
                self.device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            self.model = self.model.to(self.device)

        self.mask_id = mask_id
        tokenizer_kwargs = {}
        if self.model.config.model_type == 'Dream':
            tokenizer_kwargs['padding_side'] = 'left'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **tokenizer_kwargs)

        self.batch_size = int(batch_size)
        self.sampling_eps = 0.
        self.max_length = max_length
        self.generation_kwargs = kwargs
        self.cfg = kwargs.get('cfg_scale', 0.0)

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            tokenize_result = self.tokenizer(e["question"], padding=True)
            return {
                "question": tokenize_result["input_ids"],
                "attention_mask": tokenize_result["attention_mask"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize, batched=True, batch_size=self.batch_size)
        ds = ds.batch(batch_size=self.batch_size)
        ds = ds.with_format("torch")

        out = []
        total_time = 0
        # minimal_length = 1e10
        # total_length = 0
        # maximum_length = 0
        request_cnt = 0
        for elem in tqdm(ds, desc="Generating..."):
            prompts = elem["question"].to(self.device)
            attention_mask = elem["attention_mask"].to(self.device)
            stop_tokens_list = elem["until"]
            start_time = time.time()
            # minimal_length = min(minimal_length, prompts.shape[1])
            # total_length += prompts.shape[1] * prompts.shape[0]
            # maximum_length = max(maximum_length, prompts.shape[1])
            request_cnt += prompts.shape[0]
            generated_answers, _ = batch_generate(self.model, prompts, attention_mask, self.generation_kwargs)
            end_time = time.time()
            total_time += end_time - start_time

            for generated_answer, prompt, stop_tokens in zip(generated_answers, prompts, stop_tokens_list):
                generated_answer = self.tokenizer.decode(generated_answer[prompt.shape[0]:], skip_special_tokens=False)
                for stop_seq in stop_tokens + ['```']:
                        if stop_seq in generated_answer:
                            generated_answer = generated_answer.split(stop_seq)[0]
                # remove special tokens
                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                out.append(generated_answer)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        print(f"Total generation time: {total_time:.2f} seconds")
        print(f"Request count: {request_cnt}")
        print_statistics()
        return out

def evaluate(model: str, task: str, batch_size: int, **kwargs):
    if model == "GSAI-ML/LLaDA-8B-Base":
        model_path = "GSAI-ML/LLaDA-8B-Base"
        is_instruct = False
    elif model == "GSAI-ML/LLaDA-8B-Instruct":
        model_path = "GSAI-ML/LLaDA-8B-Instruct"
        is_instruct = True
    elif model == "Dream-org/Dream-v0-Base-7B":
        model_path = "Dream-org/Dream-v0-Base-7B"
        is_instruct = False
    elif model == "Dream-org/Dream-v0-Instruct-7B":
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        is_instruct = True
    else:
        raise ValueError(f"Not supported model: {model}")
    
    confirm_run_unsafe_code = False
    if task == "bbh":
        tasks = "bbh"
        limit = None
    elif task == "gsm8k":
        tasks = "gsm8k"
        limit = None
    elif task == "minerva_math":
        tasks = "minerva_math"
        limit = None
    elif task == "humaneval":
        tasks = "humaneval"
        limit = None
        confirm_run_unsafe_code = True
        is_instruct = False
    elif task == "mbpp":
        tasks = "mbpp"
        limit = None
        confirm_run_unsafe_code = True
    elif task == "mmlu_pro":
        tasks = "mmlu_pro"
        limit = None
    else:
        raise ValueError(f"Not supported task: {task}")
    
    if confirm_run_unsafe_code:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    output_path = f"eval_results/{task}/{model.replace('/', '__')}"

    set_seed(123)
    args = setup_parser().parse_args([])
    args.__dict__.update({
        "tasks": tasks,
        "model": "llada_dist",
        "model_args": {"model_path": model_path, **kwargs},
        "batch_size": batch_size,
        "limit": limit,
        # "output_path": output_path,
        # "log_samples": True,
        "cache_requests": True,
        "apply_chat_template": is_instruct,
        "fewshot_as_multiturn": is_instruct,
        # "num_fewshot": 0,
        "confirm_run_unsafe_code": confirm_run_unsafe_code
    })
    print(args)
    cli_evaluate(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--esdllm_mode", type=str, help="ES-dLLM mode", required=True)
    parser.add_argument("--prompt_update_freq", type=int, default=None, help="Prompt update frequency for ES-dLLM")
    parser.add_argument("--block_update_freq", type=int, default=None, help="Block update frequency for ES-dLLM")
    parser.add_argument("--alpha", type=float, default=0.5, help="Importance score alpha")
    parser.add_argument("--proportions", nargs='*', type=float, default=[], help="Proportions after skipping")
    parser.add_argument("--positions", nargs='*', type=float, default=[], help="Skipping positions")
    args = parser.parse_args()

    assert len(args.proportions) == len(args.positions)
    proportion_steps = None
    if len(args.proportions) >= 1:
        proportion_steps = [(pos, pr) for pos, pr in zip(args.proportions, args.positions)]

    if args.model == "LLaDA-Instruct":
        model_name = "GSAI-ML/LLaDA-8B-Instruct"
        model_type = "LLaDA"
    elif args.model == "LLaDA-Base":
        model_name = "GSAI-ML/LLaDA-8B-Base"
        model_type = "LLaDA"
    elif args.model == "Dream-Base":
        model_name = "Dream-org/Dream-v0-Base-7B"
        model_type = "Dream"
    elif args.model == "Dream-Instruct":
        model_name = "Dream-org/Dream-v0-Instruct-7B"
        model_type = "Dream"
    else:
        raise ValueError(f"Not supported model: {args.model}")

    # Generation length and block length binding to tasks
    if args.task == "bbh":
        batch_size = 8
        gen_length = 256
        block_length = 64
    elif args.task == "gsm8k":
        batch_size = 8
        gen_length = 256
        block_length = 64
    elif args.task == "minerva_math":
        batch_size = 8
        gen_length = 256
        block_length = 256
    elif args.task == "humaneval":
        batch_size = 8
        gen_length = 512
        block_length = 64
    elif args.task == "mbpp":
        batch_size = 8
        gen_length = 512
        block_length = 64
    elif args.task == "mmlu_pro":
        batch_size = 8
        gen_length = 256
        block_length = 256
    else:
        raise ValueError(f"Not supported task: {args.task}")

    token_per_step = 1
    threshold = None
    parallel_mode = False

    sparse_kv = 1.0
    delay_step = -1

    mode_list = args.esdllm_mode.split("_")

    for mode in mode_list[1:]:
        if mode == "p":
            threshold = 0.9
            parallel_mode = True
        elif mode == "s":
            sparse_kv = 0.5
            delay_step = 1
        else:
            raise ValueError(f"Not supported ES-dLLM mode: {args.esdllm_mode}")
    args.esdllm_mode = mode_list[0]
    
    if args.esdllm_mode in "DualCache":
        ESdLLM_mode = None
        use_kvcache = True
    elif args.esdllm_mode == "nocache":
        ESdLLM_mode = None
        use_kvcache = False
    elif args.esdllm_mode in ["HiddenState", "Key", "Value", "Query"]:
        ESdLLM_mode = args.esdllm_mode
        use_kvcache = True
        if model_type == "LLaDA":
            decode_proportions = [[p for p, s in proportion_steps if i >= s * 32][-1] for i in range(32)]
        else:
            decode_proportions = [[p for p, s in proportion_steps if i >= s * 28][-1] for i in range(28)]
        print(f"FLOPs proportions: {(sum(decode_proportions) + 1 - decode_proportions[-1]) / len(decode_proportions):.2f}")
    else:
        raise ValueError(f"Not supported ES-dLLM mode: {args.esdllm_mode}")

    generation_kwargs = {
        "gen_length": gen_length,
        "block_length": block_length,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "use_kvcache": use_kvcache,
        "parallel_mode": parallel_mode,
        "token_per_step": token_per_step,
        "threshold": threshold,
        "print_log": False,
        "record_time": False,
        "statistics": False,
        "delay_eos_generation": True,
        # Sparse attention parameters
        "sparse_kv": sparse_kv,
        "delay_step": delay_step,
        # Dream specific parameters
        "top_p": 0.95,
        "top_k": 50,
        # ES-dLLM configuration
        "block_update_freq": args.block_update_freq,
        "prompt_update_freq": args.prompt_update_freq,
        "ESdLLM_mode": ESdLLM_mode,
        "importance_score_alpha": args.alpha,
        "proportion_steps": proportion_steps,
    }

    evaluate(model_name, args.task, batch_size, **generation_kwargs)
 
    accelerate.PartialState().destroy_process_group()
