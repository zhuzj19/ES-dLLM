MACHINE_ID=$1
# LLaDA-Instruct ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh LLaDA-Instruct humaneval HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct gsm8k HiddenState ${MACHINE_ID} 64 16 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct mbpp HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct bbh HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Instruct minerva_math HiddenState ${MACHINE_ID} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

# LLaDA-Instruct DualCache
bash eval_singlegpu.sh LLaDA-Instruct humaneval DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct gsm8k DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct mbpp DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct bbh DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct minerva_math DualCache ${MACHINE_ID}

# LLaDA-Instruct nocache
bash eval_singlegpu.sh LLaDA-Instruct humaneval nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct gsm8k nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct mbpp nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct bbh nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Instruct minerva_math nocache ${MACHINE_ID}

# Dream-Instruct ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh Dream-Instruct humaneval HiddenState ${MACHINE_ID} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct gsm8k HiddenState ${MACHINE_ID} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct mbpp HiddenState ${MACHINE_ID} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct bbh HiddenState ${MACHINE_ID} 64 8 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct minerva_math HiddenState ${MACHINE_ID} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5

# Dream-Instruct DualCache
bash eval_singlegpu.sh Dream-Instruct humaneval DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct gsm8k DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct mbpp DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct bbh DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct minerva_math DualCache ${MACHINE_ID}

# Dream-Instruct nocache
bash eval_singlegpu.sh Dream-Instruct humaneval nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct gsm8k nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct mbpp nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct bbh nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Instruct minerva_math nocache ${MACHINE_ID}

# ES-dLLM*
bash eval_singlegpu.sh LLaDA-Instruct bbh HiddenState ${MACHINE_ID} 32 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct mbpp HiddenState ${MACHINE_ID} 8 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Instruct bbh HiddenState ${MACHINE_ID} 16 4 "1 0.5 0.25" "0 0.125 0.25" 0.5