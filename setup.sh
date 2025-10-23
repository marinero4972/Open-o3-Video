cd src/r1-v 
pip install -e ".[dev]"

pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install wandb==0.18.3 tensorboardx qwen_vl_utils torchvision nltk rouge_score deepspeed wheel decord openpyxl xlsxwriter pysubs2 timm
pip install flash_attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install vllm==0.7.2