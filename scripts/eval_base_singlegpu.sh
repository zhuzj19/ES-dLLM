MACHINE_ID=$1
# LLaDA-Base ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh LLaDA-Base humaneval HiddenState ${MACHINE_ID} 128 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Base mbpp HiddenState ${MACHINE_ID} 256 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Base gsm8k HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Base bbh HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh LLaDA-Base minerva_math HiddenState ${MACHINE_ID} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5

# LLaDA-Base DualCache
bash eval_singlegpu.sh LLaDA-Base humaneval DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base gsm8k DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base mbpp DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base bbh DualCache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base minerva_math DualCache ${MACHINE_ID}

# LLaDA-Base nocache
bash eval_singlegpu.sh LLaDA-Base humaneval nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base gsm8k nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base mbpp nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base bbh nocache ${MACHINE_ID}
bash eval_singlegpu.sh LLaDA-Base minerva_math nocache ${MACHINE_ID}

# Dream-Base ES-dLLM(HiddenState 0.5)
bash eval_singlegpu.sh Dream-Base humaneval HiddenState ${MACHINE_ID} 64 2 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Base gsm8k HiddenState ${MACHINE_ID} 128 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Base mbpp HiddenState ${MACHINE_ID} 256 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Base bbh HiddenState ${MACHINE_ID} 64 4 "1 0.5 0.25" "0 0.125 0.25" 0.5
bash eval_singlegpu.sh Dream-Base minerva_math HiddenState ${MACHINE_ID} 256 8 "1 0.5 0.25" "0 0.125 0.25" 0.5

# Dream-Base DualCache
bash eval_singlegpu.sh Dream-Base humaneval DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base gsm8k DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base mbpp DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base bbh DualCache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base minerva_math DualCache ${MACHINE_ID}

# Dream-Base nocache
bash eval_singlegpu.sh Dream-Base humaneval nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base gsm8k nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base mbpp nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base bbh nocache ${MACHINE_ID}
bash eval_singlegpu.sh Dream-Base minerva_math nocache ${MACHINE_ID}