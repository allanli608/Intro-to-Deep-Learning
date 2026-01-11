import os

# CONFIGURATION: Set the HF Mirror before importing transformers/hf_hub
# This tells the library to look at hf-mirror.com instead of huggingface.co
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "./Qwen2.5-1.5B-Instruct"

print(f"Downloading {model_name} from hf-mirror.com...")

# snapshot_download downloads the entire repository (weights, tokenizer, config)
path = snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Download actual files, not symlinks
    resume_download=True           # Useful if connection drops
)

print(f"\nSuccess! Model downloaded to: {path}")
print(f"You can now run your assignment with:")
print(f"python grpo_homework.py {local_dir}")