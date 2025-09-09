# get_creativity_activations.py
"""
Extract creativity-related activations from LLaMA model for ITI.
Works with the creativity dataset containing partial solutions.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')


def load_creativity_dataset(split='train'):
    """Load creativity dataset with partial solutions."""
    
    data_dir = Path('creativity_data_partial')
    
    # Try pickle first (faster)
    pkl_path = data_dir / f'{split}.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # Fallback to JSON
    json_path = data_dir / f'{split}.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    raise FileNotFoundError(f"No data found for split '{split}' in {data_dir}")


def extract_activations_with_hooks(model, tokenizer, data, device='cuda'):
    """
    Extract activations using PyTorch hooks (more robust than baukit for LLaMA 3.1).
    """
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    print(f"Model architecture: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")
    
    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_labels = []
    all_problem_ids = []
    
    for item in tqdm(data, desc="Extracting activations"):
        try:
            # Get the prompt from the data item
            prompt = item['prompt']
            label = item['label']
            problem_id = item['problem_id']
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = inputs['input_ids'].to(device)
            
            # Storage for head outputs
            head_outputs = {}
            
            # Register hooks to capture inputs to o_proj
            def get_hook(layer_idx):
                def hook(module, input, output):
                    # input[0] contains the concatenated head outputs
                    head_outputs[layer_idx] = input[0].detach()
                return hook
            
            handles = []
            for layer_idx in range(num_layers):
                handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    get_hook(layer_idx)
                )
                handles.append(handle)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # 1. Extract layer-wise hidden states (after each layer's processing)
            hidden_states = outputs.hidden_states
            layer_wise_acts = []
            for layer_idx in range(1, len(hidden_states)):  # Skip embedding layer
                last_token = hidden_states[layer_idx][0, -1, :].cpu().numpy()
                layer_wise_acts.append(last_token)
            layer_wise_acts = np.array(layer_wise_acts)
            
            # 2. Extract head-wise activations (concatenated head outputs before o_proj)
            head_wise_acts = []
            for layer_idx in range(num_layers):
                if layer_idx in head_outputs:
                    # Take last token position
                    last_token = head_outputs[layer_idx][0, -1, :].cpu().numpy()
                    head_wise_acts.append(last_token)
                else:
                    print(f"Warning: Layer {layer_idx} head outputs not captured")
                    head_wise_acts.append(np.zeros(model.config.hidden_size))
            head_wise_acts = np.array(head_wise_acts)
            
            all_layer_wise_activations.append(layer_wise_acts)
            all_head_wise_activations.append(head_wise_acts)
            all_labels.append(label)
            all_problem_ids.append(problem_id)
            
        except Exception as e:
            print(f"\nError processing problem {item.get('problem_id', 'unknown')}: {e}")
            # Add zero activations as fallback
            all_layer_wise_activations.append(
                np.zeros((num_layers, model.config.hidden_size))
            )
            all_head_wise_activations.append(
                np.zeros((num_layers, model.config.hidden_size))
            )
            all_labels.append(item.get('label', 0))
            all_problem_ids.append(item.get('problem_id', 'unknown'))
    
    # Convert to numpy arrays
    layer_wise_activations = np.array(all_layer_wise_activations)
    head_wise_activations = np.array(all_head_wise_activations)
    labels = np.array(all_labels)
    
    return layer_wise_activations, head_wise_activations, labels, all_problem_ids


def extract_activations_with_baukit(model, tokenizer, data, device='cuda'):
    """
    Alternative using baukit TraceDict (if it works with your model).
    """
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    print(f"Model architecture: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")
    
    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_labels = []
    all_problem_ids = []
    
    # Hook points - we need the INPUT to o_proj
    HOOK_POINTS = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
    
    for item in tqdm(data, desc="Extracting activations"):
        try:
            prompt = item['prompt']
            label = item['label']
            problem_id = item['problem_id']
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = inputs['input_ids'].to(device)
            
            with torch.no_grad():
                # Use TraceDict to capture activations
                with TraceDict(model, HOOK_POINTS) as td:
                    outputs = model(input_ids, output_hidden_states=True)
                
                # 1. Extract layer-wise hidden states
                hidden_states = outputs.hidden_states
                layer_wise_acts = []
                for layer_idx in range(1, len(hidden_states)):
                    last_token = hidden_states[layer_idx][0, -1, :].cpu().numpy()
                    layer_wise_acts.append(last_token)
                layer_wise_acts = np.array(layer_wise_acts)
                
                # 2. Extract head-wise activations
                head_wise_acts = []
                for layer_idx, hook_name in enumerate(HOOK_POINTS):
                    if hook_name in td:
                        # Get the INPUT to o_proj
                        o_proj_input = td[hook_name].input[0]
                        last_token_heads = o_proj_input[0, -1, :].cpu().numpy()
                        head_wise_acts.append(last_token_heads)
                    else:
                        print(f"Warning: {hook_name} not found in trace")
                        head_wise_acts.append(np.zeros(model.config.hidden_size))
                
                head_wise_acts = np.array(head_wise_acts)
                
                all_layer_wise_activations.append(layer_wise_acts)
                all_head_wise_activations.append(head_wise_acts)
                all_labels.append(label)
                all_problem_ids.append(problem_id)
                
        except Exception as e:
            print(f"\nError processing problem {item.get('problem_id', 'unknown')}: {e}")
            # Fallback
            all_layer_wise_activations.append(
                np.zeros((num_layers, model.config.hidden_size))
            )
            all_head_wise_activations.append(
                np.zeros((num_layers, model.config.hidden_size))
            )
            all_labels.append(item.get('label', 0))
            all_problem_ids.append(item.get('problem_id', 'unknown'))
    
    layer_wise_activations = np.array(all_layer_wise_activations)
    head_wise_activations = np.array(all_head_wise_activations)
    labels = np.array(all_labels)
    
    return layer_wise_activations, head_wise_activations, labels, all_problem_ids


def save_activations(layer_wise, head_wise, labels, problem_ids, 
                     num_layers, num_heads, split, save_dir):
    """Save activations in multiple formats for compatibility."""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    hidden_size = layer_wise.shape[-1]
    head_dim = hidden_size // num_heads
    
    # Save as pickle (comprehensive)
    save_dict = {
        'layer_wise': layer_wise,
        'head_wise': head_wise,
        'labels': labels,
        'problem_ids': problem_ids,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'hidden_size': hidden_size,
        'head_dim': head_dim
    }
    
    save_path_pkl = save_path / f'{split}_activations.pkl'
    with open(save_path_pkl, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"✓ Saved pickle to {save_path_pkl}")
    
    # Save as numpy arrays (for compatibility with existing code)
    save_path_layer = save_path / f'{split}_layer_wise.npy'
    save_path_head = save_path / f'{split}_head_wise.npy'
    save_path_labels = save_path / f'{split}_labels.npy'
    
    np.save(save_path_layer, layer_wise)
    np.save(save_path_head, head_wise)
    np.save(save_path_labels, labels)
    
    print(f"✓ Saved layer-wise to {save_path_layer}")
    print(f"✓ Saved head-wise to {save_path_head}")
    print(f"✓ Saved labels to {save_path_labels}")


def main():
    """Main extraction pipeline."""
    
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Change as needed
    SAVE_DIR = "features/creativity_llama31_fixed"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BAUKIT = False  # Set to False to use PyTorch hooks (more reliable for LLaMA 3.1)
    
    print(f"Loading model: {MODEL_NAME}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model config
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    print(f"Model config: {num_layers} layers, {num_heads} heads")
    print(f"Using device: {DEVICE}")
    
    # Check if data exists
    data_dir = Path('creativity_data_partial')
    if not data_dir.exists():
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("Please run create_partial_dataset.py first to generate the data.")
        return
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        try:
            # Load data
            data = load_creativity_dataset(split)
            print(f"Loaded {len(data)} samples")
            
            # Count labels
            creative_count = sum(1 for item in data if item['label'] == 1)
            print(f"  Creative: {creative_count}")
            print(f"  Non-creative: {len(data) - creative_count}")
            
            # Extract activations
            if USE_BAUKIT:
                layer_wise, head_wise, labels, problem_ids = extract_activations_with_baukit(
                    model, tokenizer, data, DEVICE
                )
            else:
                layer_wise, head_wise, labels, problem_ids = extract_activations_with_hooks(
                    model, tokenizer, data, DEVICE
                )
            
            print(f"\nActivations extracted successfully!")
            print(f"Layer-wise shape: {layer_wise.shape}")
            print(f"Head-wise shape: {head_wise.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Creative samples: {np.sum(labels)} / {len(labels)}")
            
            # Save activations
            save_activations(
                layer_wise, head_wise, labels, problem_ids,
                num_layers, num_heads, split, SAVE_DIR
            )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {split} split...")
            continue
    
    print(f"\n{'='*60}")
    print("ACTIVATION EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"All activations saved to: {SAVE_DIR}")
    print("\nNext steps:")
    print("1. Run train_creativity_probes.py to train probes")
    print("2. Run apply_creativity_iti.py to apply interventions")


if __name__ == "__main__":
    main()