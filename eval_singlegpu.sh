#!/bin/bash
# usage: bash eval_singlegpu.sh <model_name> <task> <esdllm_mode> <machine_id> (for ES-dLLM <prompt_update_freq> <block_update_freq> <proportions> <positions> <alpha>)

MODEL=$1
TASK=$2
ESDLLM_MODE=$3
MACHINE_ID=$4
PROMPT_UPDATE_FREQ="${5:-}"
BLOCK_UPDATE_FREQ="${6:-}"
PROPORTIONS="${7:-}"
POSITIONS="${8:-}"
ALPHA="${9:-}"

date_now=$(date +%s)

mkdir -p log_results/${TASK}

if [ $# -gt 4 ]; then
    CUDA_VISIBLE_DEVICES=${MACHINE_ID} python eval.py \
        --model $MODEL \
        --task $TASK \
        --esdllm_mode $ESDLLM_MODE \
        --alpha $ALPHA \
        --prompt_update_freq $PROMPT_UPDATE_FREQ \
        --block_update_freq $BLOCK_UPDATE_FREQ \
        --proportions $PROPORTIONS \
        --positions $POSITIONS \
        > log_results/${TASK}/${MODEL}_${TASK}_${ESDLLM_MODE}_${date_now}.log
else
    CUDA_VISIBLE_DEVICES=${MACHINE_ID} python eval.py \
        --model $MODEL \
        --task $TASK \
        --esdllm_mode $ESDLLM_MODE \
        > log_results/${TASK}/${MODEL}_${TASK}_${ESDLLM_MODE}_${date_now}.log
fi