NUM_PROCESSES=$1
VISIBLE_DEVICES=$2
# LLaDA-Base ES-dLLM(HiddenState 0.5)
bash eval.sh LLaDA-Base humaneval HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 128 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Base mbpp HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Base gsm8k HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Base bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Base minerva_math HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5

# LLaDA-Base DualCache
bash eval.sh LLaDA-Base humaneval DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base gsm8k DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base mbpp DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base bbh DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base minerva_math DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# LLaDA-Base nocache
bash eval.sh LLaDA-Base humaneval nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base gsm8k nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base mbpp nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base bbh nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Base minerva_math nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# Dream-Base ES-dLLM(HiddenState 0.5)
bash eval.sh Dream-Base humaneval HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Base gsm8k HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 128 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Base mbpp HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Base bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Base minerva_math HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

# Dream-Base DualCache
bash eval.sh Dream-Base humaneval DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base gsm8k DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base mbpp DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base bbh DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base minerva_math DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# Dream-Base nocache
bash eval.sh Dream-Base humaneval nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base gsm8k nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base mbpp nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base bbh nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Base minerva_math nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}