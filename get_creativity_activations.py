# get_creativity_activations.py
"""
Extract activations from LLaMA models for creativity analysis.
Uses baukit's TraceDict to get true head-wise activations from attention layers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
from baukit import TraceDict

def get_llama_activations_bau(model, prompt, device):
    """
    Extract layer-wise and head-wise activations using baukit's TraceDict.
    This follows the honest_llama implementation.
    
    Args:
        model: The LLaMA model
        prompt: Tokenized prompt tensor
        device: Device to run on (cuda/cpu)
    
    Returns:
        layer_wise_activations: Hidden states from each layer
        head_wise_activations: Activations from each attention head
    """
    # Define hooks for attention head outputs
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    
    with torch.no_grad():
        prompt = prompt.to(device)
        
        # Use TraceDict to capture attention head outputs
        with TraceDict(model, HEADS) as ret:
            output = model(prompt, output_hidden_states=True)
        
        # Get layer-wise hidden states (for each transformer layer)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        layer_wise_activations = hidden_states.detach().cpu().numpy()
        
        # Get head-wise activations from attention heads
        head_wise_hidden_states = []
        for head in HEADS:
            head_output = ret[head].output.squeeze().detach().cpu()
            head_wise_hidden_states.append(head_output)
        
        head_wise_activations = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
    
    return layer_wise_activations, head_wise_activations


def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Load model and tokenizer with proper configuration."""
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    return model, tokenizer


def format_prompt_for_generation(prompt_data):
    """
    Format the prompt for generation.
    Adjust this based on your specific prompt format needs.
    """
    # Extract the problem part before asking for completion
    if 'prompt' in prompt_data:
        problem_text = prompt_data['prompt'].split("Complete this solution:")[0].strip()
    else:
        problem_text = str(prompt_data)
    
    return problem_text


def main():
    """Main function to extract activations for creativity dataset."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get model configuration
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    print(f"Model config: {num_layers} layers, {num_heads} heads")
    print(f"Using device: {device}")
    
    # Create output directory
    save_dir = Path('features') / 'creativity_llama31_fixed'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Load creativity dataset
        with open(f'creativity_data_partial/{split}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        prompts = []
        labels = []
        problem_ids = []
        
        # Format prompts for generation
        for item in data:
            prompt_text = format_prompt_for_generation(item)
            prompts.append(prompt_text)
            labels.append(item['label'])
            problem_ids.append(item.get('problem_id', 'unknown'))
        
        print(f"Extracting activations for {len(prompts)} samples...")
        
        # Storage for activations
        all_layer_wise_activations = []
        all_head_wise_activations = []
        
        # Process each prompt
        for i, prompt_text in enumerate(tqdm(prompts, desc="Extracting activations")):
            try:
                # Tokenize the prompt
                inputs = tokenizer(
                    prompt_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                
                # Get activations using baukit
                layer_wise_acts, head_wise_acts = get_llama_activations_bau(
                    model, 
                    inputs['input_ids'], 
                    device
                )
                
                # Extract last token activations (most relevant for generation)
                # Shape: [num_layers, seq_len, hidden_dim] -> [num_layers, hidden_dim]
                last_token_layer_acts = layer_wise_acts[:, -1, :].copy()
                
                # Shape: [num_heads, seq_len, head_dim] -> [num_heads, head_dim]
                last_token_head_acts = head_wise_acts[:, -1, :].copy()
                
                all_layer_wise_activations.append(last_token_layer_acts)
                all_head_wise_activations.append(last_token_head_acts)
                
            except Exception as e:
                print(f"\nError processing prompt {i} (problem {problem_ids[i]}): {e}")
                # Add zero activations as fallback
                all_layer_wise_activations.append(
                    np.zeros((num_layers, config.hidden_size))
                )
                all_head_wise_activations.append(
                    np.zeros((num_layers, config.hidden_size))  # Note: total hidden size, not per-head
                )
        
        # Convert to numpy arrays
        layer_wise_activations = np.array(all_layer_wise_activations)
        head_wise_activations = np.array(all_head_wise_activations)
        labels = np.array(labels)
        
        print(f"\nActivations extracted successfully!")
        print(f"Layer-wise shape: {layer_wise_activations.shape}")
        print(f"Head-wise shape: {head_wise_activations.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Creative samples: {np.sum(labels)} / {len(labels)}")
        
        # Save activations in multiple formats for compatibility
        
        # 1. Save as pickle (comprehensive)
        save_dict = {
            'layer_wise': layer_wise_activations,
            'head_wise': head_wise_activations,
            'labels': labels,
            'problem_ids': problem_ids,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'hidden_size': config.hidden_size
        }
        
        save_path_pkl = save_dir / f'{split}_activations.pkl'
        with open(save_path_pkl, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"✓ Saved pickle to {save_path_pkl}")
        
        # 2. Save as numpy arrays (for compatibility with honest_llama)
        save_path_layer = save_dir / f'{split}_layer_wise.npy'
        save_path_head = save_dir / f'{split}_head_wise.npy'
        save_path_labels = save_dir / f'{split}_labels.npy'
        
        np.save(save_path_layer, layer_wise_activations)
        np.save(save_path_head, head_wise_activations)
        np.save(save_path_labels, labels)
        
        print(f"✓ Saved layer-wise to {save_path_layer}")
        print(f"✓ Saved head-wise to {save_path_head}")
        print(f"✓ Saved labels to {save_path_labels}")
    
    print(f"\n{'='*60}")
    print("ACTIVATION EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"All activations saved to: {save_dir}")
    print("\nNext steps:")
    print("1. Run train_creativity_probes.py to train probes")
    print("2. Run apply_creativity_iti.py to apply interventions")


if __name__ == "__main__":
    main()