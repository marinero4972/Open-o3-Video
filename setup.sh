cd src/r1-v 
pip install -e ".[dev]"

pip install "transformers==4.57.3"
pip install wandb==0.18.3 tensorboardx qwen_vl_utils torchvision nltk rouge_score deepspeed wheel decord openpyxl xlsxwriter pysubs2 timm
pip install flash_attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install vllm==0.11.0
