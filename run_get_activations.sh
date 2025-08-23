#!/bin/bash
# run_get_activations.sh

# Navigate to get_activations directory
cd get_activations

# Run activation collection for LLaMA 3.1 8B Instruct on TruthfulQA MC2
echo "Collecting activations for LLaMA 3.1 8B Instruct..."
CUDA_VISIBLE_DEVICES=0 python get_activations.py \
    --model_name llama3.1_8B_instruct \
    --dataset_name tqa_mc2 \
    --device 0

# Also get activations for the generation dataset (optional but recommended)
echo "Collecting activations for generation dataset..."
CUDA_VISIBLE_DEVICES=0 python get_activations.py \
    --model_name llama3.1_8B_instruct \
    --dataset_name tqa_gen \
    --device 0

echo "Activation collection complete!"