CONDA_RUN='conda run -n esdllm --no-capture-output'

conda create -n esdllm python=3.13
$CONDA_RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
$CONDA_RUN pip3 install transformers accelerate datasets
$CONDA_RUN pip3 install git+https://github.com/EleutherAI/lm-evaluation-harness@84aa9f9
