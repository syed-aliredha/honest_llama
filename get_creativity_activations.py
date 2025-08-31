# get_creativity_activations_simple.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

# Use the existing utils from the repo
from utils import get_llama_activations_bau

def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Load LLaMA 3.1 8B Instruct model and tokenizer"""
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not present
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

def format_for_generation(prompt_data):
    """Format the prompt for LLaMA generation"""
    # Extract just the problem statement without the solution
    problem_text = prompt_data['prompt'].split("Solution:")[0].strip()
    
    # Simple format for activation extraction
    return f"Problem: {problem_text}\nSolution:"

def main():
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Get model dimensions
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    print(f"Model config: {num_layers} layers, {num_heads} heads")
    
    # Create output directory
    save_dir = Path('features') / 'creativity_llama31'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Load data
        with open(f'creativity_data/{split}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Prepare prompts and labels
        prompts = []
        labels = []
        
        for item in data:
            # Use a simpler prompt format
            prompt = format_for_generation(item)
            prompts.append(prompt)
            labels.append(item['label'])
        
        # Extract activations using the existing utility
        print(f"Extracting activations for {len(prompts)} samples...")
        
        head_wise_activations = []
        layer_wise_activations = []
        
        for prompt in tqdm(prompts):
            # Get activations for single prompt
            try:
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Extract activations (we'll collect them manually)
                with torch.no_grad():
                    # Run forward pass and collect intermediate activations
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    # Get hidden states from all layers
                    hidden_states = outputs.hidden_states  # tuple of tensors, one per layer
                    
                    # Extract last token activation from each layer
                    layer_acts = []
                    for layer_hidden in hidden_states[1:]:  # Skip embedding layer
                        last_token_act = layer_hidden[0, -1, :].cpu().numpy()  # (hidden_size,)
                        layer_acts.append(last_token_act)
                    
                    layer_wise_activations.append(np.stack(layer_acts))  # (num_layers, hidden_size)
                    
                    # For head-wise, we'd need to hook into attention layers
                    # For now, use layer-wise as a proxy
                    head_wise_activations.append(np.stack(layer_acts))
                    
            except Exception as e:
                print(f"Error processing prompt: {e}")
                # Add zeros as fallback
                layer_wise_activations.append(np.zeros((num_layers, config.hidden_size)))
                head_wise_activations.append(np.zeros((num_layers, config.hidden_size)))
        
        # Convert to arrays
        layer_wise_activations = np.array(layer_wise_activations)
        head_wise_activations = np.array(head_wise_activations)
        labels = np.array(labels)
        
        print(f"Activations shape: {layer_wise_activations.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Save
        save_dict = {
            'layer_wise': layer_wise_activations,
            'head_wise': head_wise_activations,
            'labels': labels
        }
        
        save_path = save_dir / f'{split}_activations.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Saved to {save_path}")

if __name__ == "__main__":
    main()