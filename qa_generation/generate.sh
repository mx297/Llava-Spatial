#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-cpu-p
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64

# Replace this with your actual job or comman
python -m qa_generation.all_generate --split_path /home/mohamed.abouelhadid/Llava-Spatial/preprocessing/preprocess_scannet/splits/scannetv2_train.txt 
                                    --split_type train --processed_data_path /l/users/mohaned.abouelhadid/sampled 
                                    --output_dir /l/users/mohamed.abouelhadid/questions --dataset scannet 
                                    --num_workers 64
