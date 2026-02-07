#!/bin/bash
# usage: bash eval.sh <model_name> <task> <esdllm_mode> <num_processes> <visible_devices> (for ES-dLLM <prompt_update_freq> <block_update_freq> <proportions> <positions> <alpha>)

MODEL=$1
TASK=$2
ESDLLM_MODE=$3
PROMPT_UPDATE_FREQ="${6:-}"
BLOCK_UPDATE_FREQ="${7:-}"
PROPORTIONS="${8:-}"
POSITIONS="${9:-}"
ALPHA="${10:-}"

date_now=$(date +%s)

mkdir -p log_results/${TASK}

if [ $# -gt 5 ]; then
    CUDA_VISIBLE_DEVICES=$5 accelerate launch --num_processes $4 eval.py \
        --model $MODEL \
        --task $TASK \
        --esdllm_mode $ESDLLM_MODE \
        --alpha $ALPHA \
        --prompt_update_freq $PROMPT_UPDATE_FREQ \
        --block_update_freq $BLOCK_UPDATE_FREQ \
        --proportions $PROPORTIONS \
        --positions $POSITIONS \
        > log_results/${TASK}/${MODEL}_${TASK}_${ESDLLM_MODE}_${date_now}_acc.log
else
    CUDA_VISIBLE_DEVICES=$5 accelerate launch --num_processes $4 eval.py \
        --model $MODEL \
        --task $TASK \
        --esdllm_mode $ESDLLM_MODE \
        > log_results/${TASK}/${MODEL}_${TASK}_${ESDLLM_MODE}_${date_now}_acc.log
fi