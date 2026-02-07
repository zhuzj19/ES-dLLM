NUM_PROCESSES=$1
VISIBLE_DEVICES=$2
# LLaDA-Instruct ES-dLLM(HiddenState 0.5)
bash eval.sh LLaDA-Instruct humaneval HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct gsm8k HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 16 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct mbpp HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct minerva_math HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

# LLaDA-Instruct DualCache
bash eval.sh LLaDA-Instruct humaneval DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct gsm8k DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct mbpp DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct bbh DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct minerva_math DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# LLaDA-Instruct nocache
bash eval.sh LLaDA-Instruct humaneval nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct gsm8k nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct mbpp nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct bbh nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct minerva_math nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# Dream-Instruct ES-dLLM(HiddenState 0.5)
bash eval.sh Dream-Instruct humaneval HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct gsm8k HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct mbpp HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct minerva_math HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5

# Dream-Instruct DualCache
bash eval.sh Dream-Instruct humaneval DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct gsm8k DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct mbpp DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct bbh DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct minerva_math DualCache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# Dream-Instruct nocache
bash eval.sh Dream-Instruct humaneval nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct gsm8k nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct mbpp nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct bbh nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct minerva_math nocache ${NUM_PROCESSES} ${VISIBLE_DEVICES}

# ES-dLLM*
bash eval.sh LLaDA-Instruct bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 32 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct mbpp HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 8 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct bbh HiddenState ${NUM_PROCESSES} ${VISIBLE_DEVICES} 16 4 "1 0.5 0.25" "0 0.125 0.25" 0.5