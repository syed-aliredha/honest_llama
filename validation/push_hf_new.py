# push_to_huggingface.py
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json

def push_to_huggingface(local_path, repo_name, private=False):
    """
    Push the honest model to HuggingFace Hub
    """
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Get your HuggingFace username
    whoami = api.whoami()
    username = whoami['name']
    
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"Preparing to push to {full_repo_name}...")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=full_repo_name, private=private, repo_type="model")
        print(f"✓ Created repository: {full_repo_name}")
    except Exception as e:
        print(f"Repository may already exist or error occurred: {e}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    
    # Push to hub
    print("Pushing model to HuggingFace Hub...")
    model.push_to_hub(full_repo_name, private=private)
    tokenizer.push_to_hub(full_repo_name, private=private)
    
    # Create and push model card
    print("Creating model card...")
    
    # Load intervention metadata if exists
    metadata_path = os.path.join(local_path, "intervention_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    model_card = f"""---
license: llama3.1
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- llama
- llama-3
- truthfulness
- ITI
- inference-time-intervention
datasets:
- truthfulqa
language:
- en
---

# Honest LLaMA 3.1 8B Instruct

This model is a LLaMA 3.1 8B Instruct model with **Inference-Time Intervention (ITI)** applied to improve truthfulness.

## Model Description

This model has been modified using the Inference-Time Intervention (ITI) technique from the paper:
["Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"](https://arxiv.org/abs/2306.03341)

### Intervention Details

- **Base Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Intervention Type**: ITI (Inference-Time Intervention)
- **Alpha (intervention strength)**: {metadata.get('alpha', 5.3)}
- **Number of heads intervened**: {metadata.get('num_heads_intervened', 48)}
- **Layers modified**: {len(metadata.get('layers_modified', []))}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "{full_repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Ask a question that typically triggers misconceptions
question = "What happens if you eat watermelon seeds?"

# Format for LLaMA 3.1 Instruct
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{{question}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids.cuda(),
    max_new_tokens=200,
    do_sample=False
)

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
"""