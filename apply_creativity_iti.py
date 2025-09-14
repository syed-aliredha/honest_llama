import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pickle
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class CreativityITI:
    
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", alpha=1.0):

        self.model_name = model_name
        self.alpha = alpha
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            do_sample=False
        )
        self.model.eval()
        
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        self.load_iti_components()
        
        self.hooks = []
        
    def load_iti_components(self):
        
        component_dir = Path('creativity_iti_components')
        
        with open(component_dir / 'top_heads.pkl', 'rb') as f:
            self.top_heads = pickle.load(f)
        
        with open(component_dir / 'directions.pkl', 'rb') as f:
            self.directions = pickle.load(f)
        
        print(f"Loaded {len(self.top_heads)} intervention heads")
        
        self.heads_by_layer = {}
        for head_info in self.top_heads:
            layer = head_info['layer']
            head = head_info['head']
            if layer not in self.heads_by_layer:
                self.heads_by_layer[layer] = []
            self.heads_by_layer[layer].append(head)
    
    def create_intervention_hook(self, layer_idx: int, head_indices: List[int]):
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            batch_size, seq_len, _ = hidden_states.shape
            
            hidden_reshaped = hidden_states.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            
            for head_idx in head_indices:
                if (layer_idx, head_idx) in self.directions:
                    direction = self.directions[(layer_idx, head_idx)]
                    direction_tensor = torch.tensor(
                        direction, 
                        dtype=hidden_reshaped.dtype,
                        device=hidden_reshaped.device
                    )
                    
                    hidden_reshaped[:, -1, head_idx, :] += self.alpha * direction_tensor
            
            hidden_states_new = hidden_reshaped.view(batch_size, seq_len, self.hidden_size)
            
            if isinstance(output, tuple):
                return (hidden_states_new,) + output[1:]
            else:
                return hidden_states_new
        
        return hook_fn
    
    def register_hooks(self):
        
        self.remove_hooks()
        
        for layer_idx, head_indices in self.heads_by_layer.items():
            hook_fn = self.create_intervention_hook(layer_idx, head_indices)
            hook = self.model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        print(f"Registered hooks on {len(self.heads_by_layer)} layers")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_creative_solution(self, problem: str, max_length=512, temperature=0.8):

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a creative Python programmer. Generate innovative and elegant solutions to coding problems.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{problem}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here's a creative Python solution:

```python"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        self.register_hooks()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=False,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        self.remove_hooks()
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "```python" in generated:
            code_start = generated.find("```python") + len("```python")
            code_end = generated.find("```", code_start)
            if code_end > code_start:
                return generated[code_start:code_end].strip()
        
        return generated
    
    def compare_with_baseline(self, problem: str):
        print("=" * 60)
        print("BASELINE SOLUTION (No Intervention)")
        print("=" * 60)
        
        self.alpha = 0
        baseline = self.generate_creative_solution(problem, temperature=0.7)
        print(baseline)
        
        print("\n" + "=" * 60)
        print(f"CREATIVE SOLUTION (ITI, α={self.original_alpha})")
        print("=" * 60)
        
        self.alpha = self.original_alpha
        creative = self.generate_creative_solution(problem, temperature=0.8)
        print(creative)
        
        return baseline, creative

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=15.0, 
                       help='Intervention strength (0=none, 15=moderate, 30=strong)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Problem to solve (optional)')
    args = parser.parse_args()
    
    iti = CreativityITI(alpha=args.alpha)
    iti.original_alpha = args.alpha
    
    test_problems = [
        "Write a function to find all unique pairs of numbers in a list that sum to a target value.",
        "Create a function that generates the Fibonacci sequence using an innovative approach.",
        "Implement a function to check if a string is a palindrome, but use an unconventional method.",
        "Write a function to flatten a nested list of arbitrary depth.",
    ]
    
    if args.problem:
        test_problems = [args.problem]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'#' * 60}")
        print(f"PROBLEM {i}: {problem}")
        print(f"{'#' * 60}")
        
        baseline, creative = iti.compare_with_baseline(problem)
        
        print("\n" + "=" * 60)
        print("CREATIVITY ANALYSIS")
        print("=" * 60)
        
        baseline_constructs = set()
        creative_constructs = set()
        
        keywords = ['for', 'while', 'if', 'else', 'elif', 'try', 'except', 
                   'lambda', 'yield', 'comprehension', 'map', 'filter', 'reduce',
                   'zip', 'enumerate', 'itertools', 'collections', 'functools']
        
        for keyword in keywords:
            if keyword in baseline:
                baseline_constructs.add(keyword)
            if keyword in creative:
                creative_constructs.add(keyword)
        
        print(f"Baseline uses: {baseline_constructs}")
        print(f"Creative uses: {creative_constructs}")
        print(f"New techniques in creative: {creative_constructs - baseline_constructs}")
        
        input("\nPress Enter for next problem...")

if __name__ == "__main__":
    main()