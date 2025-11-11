#!/bin/bash
#SBATCH --job-name=llava3_test
#SBATCH --time=01:00:00              # 1 hour max runtime
#SBATCH --nodes=1                    # 1 node
#SBATCH -p long                      # GPU partition
#SBATCH -q gpu-12                    # QOS for GPU jobs
#SBATCH --gres=gpu:4                 # Request 4 A100 GPUs
#SBATCH --mem=230G                   # System memory
#SBATCH --cpus-per-task=64           # CPU cores for data loading
#SBATCH --ntasks-per-node=1          # One training process
#SBATCH -o /l/users/rana.zayed/projects/Llava-Spatial/test_llama3/logs/llava3_test_%j.out   # Save stdout/stderr in logs/


# --- Environment setup ---
source $HOME/.bashrc
cd /l/users/rana.zayed/projects/Llava-Spatial

# Activate venv
source .venv/bin/activate

# Allow venv to see user-installed packages (optional safety)
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# --- Diagnostics ---
echo "=== Allocated GPUs ==="
nvidia-smi
echo "=== Python Path ==="
which python
python --version
# --- Run your small test job ---

python -m llava.train.train_spatial \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --vision_tower openai/clip-vit-base-patch32 \
  --data_path /l/users/rana.zayed/projects/Llava-Spatial/test_llama3/sample_train.json \
  --image_folder /l/users/rana.zayed/projects/Llava-Spatial/test_llama3/images \
  --bf16 True \
  --output_dir /l/users/rana.zayed/projects/Llava-Spatial/test_llama3/output_test \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --model_max_length 512 \
  --save_steps 10 \
  --logging_steps 1 \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --vision_tower_pretrained openai/clip-vit-base-patch32

