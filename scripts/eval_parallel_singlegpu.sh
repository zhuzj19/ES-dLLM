MACHINE_ID=$1
# LLaDA-Instruct ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh LLaDA-Instruct humaneval HiddenState_p ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct gsm8k HiddenState_p ${MACHINE_ID} 64 16 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct mbpp HiddenState_p ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct bbh HiddenState_p ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct minerva_math HiddenState_p ${MACHINE_ID} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

# LLaDA-Instruct DualCache
bash eval_singlegpu.sh LLaDA-Instruct humaneval DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct gsm8k DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct mbpp DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct bbh DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct minerva_math DualCache_p ${MACHINE_ID}

# Dream-Instruct ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh Dream-Instruct humaneval HiddenState_p ${MACHINE_ID} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct gsm8k HiddenState_p ${MACHINE_ID} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct mbpp HiddenState_p ${MACHINE_ID} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct bbh HiddenState_p ${MACHINE_ID} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct minerva_math HiddenState_p ${MACHINE_ID} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5

# Dream-Instruct DualCache
bash eval_singlegpu.sh Dream-Instruct humaneval DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct gsm8k DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct mbpp DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct bbh DualCache_p ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct minerva_math DualCache_p ${MACHINE_ID}