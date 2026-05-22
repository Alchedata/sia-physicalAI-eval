from huggingface_hub import create_repo, HfApi

# 1. create repository
hf_name = "AlphaBrain/Qwen3-VL-OFT-LIBERO-4in1"
create_repo(hf_name, repo_type="model", exist_ok=True)

# 2. initialize API
api = HfApi()

# 3. upload large folder
import os as _os
folder_path = _os.environ.get(
    "UPLOAD_FOLDER_PATH",
    "./results/training/Qwen3-VL-OFT-LIBERO-4in1",  # TODO: set UPLOAD_FOLDER_PATH env var before running
)
api.upload_large_folder(folder_path=folder_path, repo_id=hf_name, repo_type="model")
