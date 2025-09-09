# fresh_huggingface_upload.py
"""
Complete fresh upload to HuggingFace - full model with ITI components
Assumes starting from scratch with no existing repositories
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo, whoami
import pickle
import json
import shutil
from pathlib import Path
import argparse

def prepare_full_model_with_iti(use_top_n_heads=48, alpha=0.35):
    """
    Prepare the full model with ITI components for upload
    """
    
    print("="*70)
    print("PREPARING FULL MODEL WITH CREATIVITY ITI")
    print("="*70)
    print(f"Configuration:")
    print(f"  Alpha: {alpha}")
    print(f"  Top heads: {use_top_n_heads}")
    print()
    
    # Create output directory
    output_dir = Path("llama-creativity-iti-upload")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    # Step 1: Load and save the base model
    print("Step 1: Loading LLaMA 3.1 8B model...")
    print("This will download ~16GB if not cached...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Step 2: Process ITI components
    print("\nStep 2: Processing ITI components...")
    
    # Load components
    with open('creativity_iti_components/top_heads.pkl', 'rb') as f:
        all_top_heads = pickle.load(f)
    
    with open('creativity_iti_components/directions.pkl', 'rb') as f:
        all_directions = pickle.load(f)
    
    # Filter to top N heads
    top_heads = all_top_heads[:use_top_n_heads]
    
    # Filter directions
    filtered_directions = {}
    for head_info in top_heads:
        layer = head_info['layer']
        head = head_info['head']
        key = (layer, head)
        if key in all_directions:
            filtered_directions[key] = all_directions[key]
    
    print(f"  Using {len(top_heads)} heads from {len(all_top_heads)} available")
    print(f"  Filtered directions to {len(filtered_directions)} entries")
    
    # Save ITI components
    with open(output_dir / "iti_top_heads.pkl", 'wb') as f:
        pickle.dump(top_heads, f)
    
    with open(output_dir / "iti_directions.pkl", 'wb') as f:
        pickle.dump(filtered_directions, f)
    
    # Create ITI config
    iti_config = {
        "alpha": alpha,
        "num_intervention_heads": len(top_heads),
        "total_heads_trained": len(all_top_heads),
        "intervention_type": "creativity",
        "auto_apply": True,
        "dataset": "NeoCoder",
        "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "training_method": "activations_from_solutions"
    }
    
    with open(output_dir / "iti_config.json", 'w') as f:
        json.dump(iti_config, f, indent=2)
    
    # Step 3: Create auto-apply wrapper
    print("\nStep 3: Creating auto-apply wrapper...")
    
    modeling_code = '''# modeling_creativity_iti.py
"""
Auto-apply creativity ITI wrapper for LLaMA 3.1 8B
"""

import torch
import pickle
import json
from pathlib import Path
from transformers import LlamaForCausalLM
from huggingface_hub import hf_hub_download

class CreativityITILlamaForCausalLM(LlamaForCausalLM):
    """LLaMA with automatic creativity ITI application"""
    
    def __init__(self, config):
        super().__init__(config)
        
        try:
            # Get model name from config
            model_name = getattr(config, "_name_or_path", "")
            
            # Download ITI files
            print(f"Loading Creativity ITI components...")
            
            top_heads_path = hf_hub_download(
                repo_id=model_name,
                filename="iti_top_heads.pkl",
                repo_type="model"
            )
            
            directions_path = hf_hub_download(
                repo_id=model_name,
                filename="iti_directions.pkl",
                repo_type="model"
            )
            
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="iti_config.json",
                repo_type="model"
            )
            
            # Load files
            with open(top_heads_path, 'rb') as f:
                self.top_heads = pickle.load(f)
            
            with open(directions_path, 'rb') as f:
                self.directions = pickle.load(f)
            
            with open(config_path, 'r') as f:
                iti_config = json.load(f)
                self.alpha = iti_config['alpha']
            
            # Model dimensions
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // self.num_heads
            
            # Register hooks
            self._register_iti_hooks()
            print(f"✓ Creativity ITI active: α={self.alpha}, {len(self.top_heads)} heads")
            
        except Exception as e:
            print(f"Warning: Could not load ITI: {e}")
            self.top_heads = []
            self.directions = {}
            self.alpha = 0
    
    def _register_iti_hooks(self):
        """Register ITI intervention hooks"""
        if not self.top_heads:
            return
            
        heads_by_layer = {}
        for head_info in self.top_heads:
            layer = head_info['layer']
            head = head_info['head']
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)
        
        for layer_idx, head_indices in heads_by_layer.items():
            def make_hook(layer_idx, head_indices):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    
                    hidden_reshaped = hidden_states.view(
                        batch_size, seq_len, self.num_heads, self.head_dim
                    )
                    
                    for head_idx in head_indices:
                        if (layer_idx, head_idx) in self.directions:
                            direction = torch.tensor(
                                self.directions[(layer_idx, head_idx)],
                                dtype=hidden_reshaped.dtype,
                                device=hidden_reshaped.device
                            )
                            
                            hidden_reshaped[:, -1, head_idx, :] += self.alpha * direction
                    
                    hidden_states = hidden_reshaped.view(batch_size, seq_len, hidden_size)
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
                
                return hook_fn
            
            hook = make_hook(layer_idx, head_indices)
            self.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(hook)
'''
    
    with open(output_dir / "modeling_creativity_iti.py", 'w') as f:
        f.write(modeling_code)
    
    # Update config.json for custom model
    config_path = output_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config["auto_map"] = {
        "AutoModelForCausalLM": "modeling_creativity_iti.CreativityITILlamaForCausalLM"
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Step 4: Create README
    print("\nStep 4: Creating documentation...")
    
    readme = f"""---
license: apache-2.0
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- text-generation
- code-generation
- creativity
- inference-time-intervention
library_name: transformers
pipeline_tag: text-generation
---

# LLaMA 3.1 8B with Creativity ITI

Full model with automatic creativity enhancement through Inference-Time Intervention.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model - ITI automatically applies!
model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/llama-31-8b-creativity-iti",
    trust_remote_code=True,  # Required for auto-ITI
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/llama-31-8b-creativity-iti")

# Generate creative code
prompt = "Write a function to check if a number is prime"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Configuration

- **Alpha**: {alpha}
- **Active Heads**: {use_top_n_heads}
- **Base Model**: LLaMA 3.1 8B Instruct
- **Intervention**: Automatic during inference

## How It Works

The model automatically applies Inference-Time Intervention to enhance creativity:
1. Monitors {use_top_n_heads} attention heads during generation
2. Shifts activations by α={alpha} toward creative directions
3. Results in more innovative code solutions

## Training

- Dataset: NeoCoder (1058 problems)
- Method: Extracted activations from complete solutions
- Metric: Novel technique usage vs human solutions

## License

Apache 2.0
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)
    
    print(f"\n✓ Model prepared at {output_dir}")
    print(f"  Total size: ~{sum(f.stat().st_size for f in output_dir.glob('*')) / 1e9:.1f} GB")
    
    return output_dir

def upload_to_huggingface(model_dir, repo_name, organization=None, private=False):
    """
    Upload the prepared model to HuggingFace
    """
    
    print("\n" + "="*70)
    print("UPLOADING TO HUGGINGFACE")
    print("="*70)
    
    api = HfApi()
    
    # Determine repo ID
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        # Get username
        user_info = whoami()
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
    
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"Size: ~16GB")
    
    response = input("\nProceed with upload? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return
    
    try:
        # Create repository
        print("\nCreating repository...")
        url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"Repository: {url}")
        
        # Upload
        print("\nUploading (this will take 10-30 minutes)...")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload LLaMA 3.1 8B with Creativity ITI"
        )
        
        print(f"\n✓ Upload complete!")
        print(f"Model available at: https://huggingface.co/{repo_id}")
        
        print("\n" + "="*70)
        print("USAGE INSTRUCTIONS")
        print("="*70)
        print(f"""
# Install dependencies
pip install transformers torch

# Use the model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate with automatic creativity enhancement!
""")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload LLaMA with Creativity ITI to HuggingFace")
    parser.add_argument("--repo-name", type=str, default="llama-31-8b-creativity-it-80-percent",
                       help="Repository name")
    parser.add_argument("--organization", type=str, default=None,
                       help="HuggingFace organization (optional)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    parser.add_argument("--alpha", type=float, default=0.35,
                       help="ITI intervention strength")
    parser.add_argument("--num-heads", type=int, default=48,
                       help="Number of top heads to use")
    parser.add_argument("--skip-upload", action="store_true",
                       help="Only prepare, don't upload")
    
    args = parser.parse_args()
    
    # Prepare model
    model_dir = prepare_full_model_with_iti(
        use_top_n_heads=args.num_heads,
        alpha=args.alpha
    )
    
    # Upload unless skipped
    if not args.skip_upload:
        upload_to_huggingface(
            model_dir=model_dir,
            repo_name=args.repo_name,
            organization=args.organization,
            private=args.private
        )
    else:
        print(f"\nModel prepared at: {model_dir}")
        print("Run without --skip-upload to upload to HuggingFace")

if __name__ == "__main__":
    main()