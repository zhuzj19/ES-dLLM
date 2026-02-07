# ES-dLLM

This repository contains the implementation of the paper "ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping". The code is built on top of [LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/DreamLM/Dream).

## Requirements

We provide a `environment.sh` script for setting up the environment. For the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) library, we use the version of commit `84aa9f9`. Please use `conda activate esdllm` to activate the environment before running test scripts.

## Usage

We provide ready-to-use scripts to reproduce the results reported in the paper:
* Main Results of Instruct models: `scripts/eval_instruct.sh`
* Main Results of Base models: `scripts/eval_base.sh`
* Integration with Parallel decoding: `scripts/eval_parallel.sh`
* Integration with Sparse-dLLM: `scripts/eval_sparse.sh`
* Integration with both Parallel decoding and Sparse-dLLM: `scripts/eval_parallel_sparse.sh`

Example (multi-GPU with 4 GPUs):
```bash
bash scripts/eval_instruct.sh 4 0,1,2,3
bash scripts/eval_base.sh 4 0,1,2,3
bash scripts/eval_parallel.sh 4 0,1,2,3
bash scripts/eval_sparse.sh 4 0,1,2,3
bash scripts/eval_parallel_sparse.sh 4 0,1,2,3
```

We used a single GPU for evaluation in the paper. The results can be slightly different from the multi-GPU due to batching order (inducing different padding lengths and therefore different kernel selections), so we also provide single-GPU version of the scripts in `scripts` folder.
```bash
bash scripts/eval_instruct_singlegpu.sh 0
bash scripts/eval_base_singlegpu.sh 0
bash scripts/eval_parallel_singlegpu.sh 0
bash scripts/eval_sparse_singlegpu.sh 0
bash scripts/eval_parallel_sparse_singlegpu.sh 0
```

Additionally, we provide general-purpose script `eval.sh` and `eval_singlegpu.sh` that can be customized for different settings (e.g., model name, ES-dLLM configuration, etc.). Please refer to the comments in the script for usage details.