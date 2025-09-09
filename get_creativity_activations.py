import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

from utils import get_llama_activations_bau

def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
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

def format_for_generation(prompt_data):
    problem_text = prompt_data['prompt'].split("Solution:")[0].strip()
    
    return f"Problem: {problem_text}\nSolution:"

def main():
    model, tokenizer = load_model_and_tokenizer()
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    print(f"Model config: {num_layers} layers, {num_heads} heads")
    
    save_dir = Path('features') / 'creativity_llama31'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        with open(f'creativity_data_partial/{split}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        prompts = []
        labels = []
        
        for item in data:
            prompt = format_for_generation(item)
            prompts.append(prompt)
            labels.append(item['label'])
        
        print(f"Extracting activations for {len(prompts)} samples...")
        
        head_wise_activations = []
        layer_wise_activations = []
        
        for prompt in tqdm(prompts):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden_states = outputs.hidden_states
                    
                    layer_acts = []
                    for layer_hidden in hidden_states[1:]:
                        last_token_act = layer_hidden[0, -1, :].cpu().numpy()
                        layer_acts.append(last_token_act)
                    
                    layer_wise_activations.append(np.stack(layer_acts))
                    head_wise_activations.append(np.stack(layer_acts))
                    
            except Exception as e:
                print(f"Error processing prompt: {e}")
                layer_wise_activations.append(np.zeros((num_layers, config.hidden_size)))
                head_wise_activations.append(np.zeros((num_layers, config.hidden_size)))
        
        layer_wise_activations = np.array(layer_wise_activations)
        head_wise_activations = np.array(head_wise_activations)
        labels = np.array(labels)
        
        print(f"Activations shape: {layer_wise_activations.shape}")
        print(f"Labels shape: {labels.shape}")
        
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