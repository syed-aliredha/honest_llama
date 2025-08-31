# flexible_iti_test.py
"""
Flexible testing script with adjustable alpha and head selection
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pickle
import argparse

class FlexibleCreativityITI:
    """Wrapper to apply ITI with custom alpha and head selection"""
    
    def __init__(
        self,
        model_name="syed-aliredha/llama-31-8b-creativity-iti-full",
        alpha=None,  # None = use default from HF, or specify custom value
        num_heads=None,  # None = use all, or specify number to use
        head_indices=None,  # None = use top heads, or specify specific indices
        device="cuda"
    ):
        print(f"Loading model with custom ITI settings...")
        
        # Load base model WITHOUT custom code to avoid automatic ITI
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,  # Disable automatic ITI
            torch_dtype=torch.float16,
            device_map="auto",
            do_sample=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Download ITI components
        print("Downloading ITI components...")
        top_heads_path = hf_hub_download(model_name, "iti_top_heads.pkl", repo_type="model")
        directions_path = hf_hub_download(model_name, "iti_directions.pkl", repo_type="model")
        
        with open(top_heads_path, 'rb') as f:
            self.all_top_heads = pickle.load(f)
        
        with open(directions_path, 'rb') as f:
            self.directions = pickle.load(f)
        
        # Model config
        config = self.model.config
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        
        # Select heads to use
        if head_indices is not None:
            # Use specific head indices
            self.top_heads = [self.all_top_heads[i] for i in head_indices if i < len(self.all_top_heads)]
        elif num_heads is not None:
            # Use top N heads
            self.top_heads = self.all_top_heads[:num_heads]
        else:
            # Use all heads
            self.top_heads = self.all_top_heads
        
        # Set alpha (default to 0.4 if not specified)
        self.alpha = alpha if alpha is not None else 0.4
        
        print(f"✓ Loaded with custom settings:")
        print(f"  Alpha: {self.alpha}")
        print(f"  Active heads: {len(self.top_heads)} / {len(self.all_top_heads)}")
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register intervention hooks with current settings"""
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Group heads by layer
        heads_by_layer = {}
        for head_info in self.top_heads:
            layer = head_info['layer']
            head = head_info['head']
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)
        
        # Register new hooks
        for layer_idx, head_indices in heads_by_layer.items():
            def make_hook(layer_idx, head_indices):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    
                    # Reshape for head-wise intervention
                    hidden_reshaped = hidden_states.view(
                        batch_size, seq_len, self.num_attention_heads, self.head_dim
                    )
                    
                    # Apply intervention
                    for head_idx in head_indices:
                        if (layer_idx, head_idx) in self.directions:
                            direction = torch.tensor(
                                self.directions[(layer_idx, head_idx)],
                                dtype=hidden_reshaped.dtype,
                                device=hidden_reshaped.device
                            )
                            
                            # Apply to last token with current alpha
                            hidden_reshaped[:, -1, head_idx, :] += self.alpha * direction
                    
                    # Reshape back
                    hidden_states = hidden_reshaped.view(batch_size, seq_len, hidden_size)
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
                
                return hook_fn
            
            hook = make_hook(layer_idx, head_indices)
            self.model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(hook)
            self.hooks.append(hook)
    
    def update_parameters(self, alpha=None, num_heads=None):
        """Update ITI parameters on the fly"""
        if alpha is not None:
            self.alpha = alpha
            print(f"Updated alpha to {alpha}")
        
        if num_heads is not None:
            self.top_heads = self.all_top_heads[:num_heads]
            print(f"Updated to use top {num_heads} heads")
        
        # Re-register hooks with new settings
        self._register_hooks()
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.8):
        """Generate with current ITI settings"""
        
        # Format prompt
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```python"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "```python" in generated:
            code_start = generated.find("```python") + len("```python")
            code_end = generated.find("```", code_start)
            if code_end > code_start:
                return generated[code_start:code_end].strip()
        
        return generated.split(prompt)[-1].strip() if prompt in generated else generated

def compare_alphas(prompt="Write a function to check if two strings are anagrams of each other."):
    """Compare different alpha values"""
    
    print("="*70)
    print("COMPARING DIFFERENT ALPHA VALUES")
    print("="*70)
    print(f"Prompt: {prompt}\n")
    
    # Test different alphas
    alphas = [0, 0.2, 0.4, 0.8, 1.5]
    
    # Initialize model once
    model = FlexibleCreativityITI(alpha=0)
    
    for alpha in alphas:
        model.update_parameters(alpha=alpha)
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")
        print(f"{'='*50}")
        
        solution = model.generate(prompt)
        print(solution)
        
        # Check for creative indicators
        indicators = ["comprehension" in solution, "lambda" in solution, 
                     "all(" in solution, len(solution.split('\n')) == 1]
        creativity_score = sum(indicators)
        print(f"\nCreativity indicators: {creativity_score}/4")

def compare_head_counts(prompt="Write a function to find Fibonacci numbers"):
    """Compare different numbers of intervention heads"""
    
    print("="*70)
    print("COMPARING DIFFERENT HEAD COUNTS")
    print("="*70)
    print(f"Prompt: {prompt}\n")
    
    # Test different head counts
    head_counts = [0, 10, 20, 48]
    
    model = FlexibleCreativityITI(alpha=0.4, num_heads=0)
    
    for num_heads in head_counts:
        if num_heads == 0:
            model.update_parameters(alpha=0)  # No intervention
            print(f"\n{'='*50}")
            print(f"No intervention (baseline)")
        else:
            model.update_parameters(alpha=0.4, num_heads=num_heads)
            print(f"\n{'='*50}")
            print(f"Using top {num_heads} heads")
        print(f"{'='*50}")
        
        solution = model.generate(prompt)
        print(solution)

def main():
    parser = argparse.ArgumentParser(description="Test ITI with custom parameters")
    parser.add_argument("--alpha", type=float, default=0.4, help="Intervention strength")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of top heads to use")
    parser.add_argument("--prompt", type=str, default="Write a function to check if a string is a palindrome")
    parser.add_argument("--compare-alphas", action="store_true", help="Compare different alpha values")
    parser.add_argument("--compare-heads", action="store_true", help="Compare different head counts")
    
    args = parser.parse_args()
    
    if args.compare_alphas:
        compare_alphas(args.prompt)
    elif args.compare_heads:
        compare_head_counts(args.prompt)
    else:
        # Single generation with custom parameters
        print(f"Generating with alpha={args.alpha}, heads={args.num_heads or 'all'}")
        
        model = FlexibleCreativityITI(
            alpha=args.alpha,
            num_heads=args.num_heads
        )
        
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerated solution:")
        print("-"*50)
        solution = model.generate(args.prompt)
        print(solution)
        print("-"*50)

if __name__ == "__main__":
    main()