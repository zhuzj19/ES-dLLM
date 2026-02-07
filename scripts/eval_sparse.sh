MACHINE_ID=$1
VISIBLE_DEVICES=$2

bash eval.sh LLaDA-Instruct humaneval DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct gsm8k DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct mbpp DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct bbh DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh LLaDA-Instruct minerva_math DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}

bash eval.sh LLaDA-Instruct humaneval HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct gsm8k HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct mbpp HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct bbh HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh LLaDA-Instruct minerva_math HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

bash eval.sh Dream-Instruct humaneval DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct gsm8k DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct mbpp DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct bbh DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}
bash eval.sh Dream-Instruct minerva_math DualCache_s ${NUM_PROCESSES} ${VISIBLE_DEVICES}

bash eval.sh Dream-Instruct humaneval HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct gsm8k HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct mbpp HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct bbh HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval.sh Dream-Instruct minerva_math HiddenState_s ${NUM_PROCESSES} ${VISIBLE_DEVICES} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5