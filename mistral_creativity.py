#!/usr/bin/env python3
"""
ITI Creativity Analysis for Mistral-7B-Instruct-v0.3
Adapted from the original LLaMA implementation
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Configuration for Mistral
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SAVE_DIR = "features/creativity_mistral_7b"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1

print("="*60)
print("MISTRAL-7B CREATIVITY ITI ANALYSIS")
print("="*60)

# ================ DATA LOADING ================

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


# ================ ACTIVATION EXTRACTION ================

def extract_activations_mistral(model, tokenizer, data, device='cuda'):
    """Extract activations from Mistral model using hooks."""
    
    model.eval()
    
    # Get model dimensions
    config = model.config
    num_layers = config.num_hidden_layers  # 32 for Mistral-7B
    num_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads  # 8 KV heads for Mistral
    hidden_size = config.hidden_size  # 4096
    head_dim = hidden_size // config.num_attention_heads  # 128
    
    print(f"Model config: {num_layers} layers, {num_heads} KV heads, {hidden_size} hidden size")
    
    # Initialize storage
    all_layer_acts = []
    all_head_acts = []
    all_labels = []
    all_problem_ids = []
    
    # Storage for hook outputs
    layer_outputs = {}
    
    def create_hook(layer_idx):
        def hook_fn(module, input, output):
            # For Mistral, output is a tuple (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            layer_outputs[layer_idx] = hidden_states.detach().cpu()
        return hook_fn
    
    # Register hooks
    hooks = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        hook = layer.self_attn.register_forward_hook(create_hook(i))
        hooks.append(hook)
    
    # Process each sample
    for item in tqdm(data, desc="Extracting activations"):
        layer_outputs.clear()
        
        # Prepare input
        text = item['prompt']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of tensors
        
        # Extract activations from last token
        layer_acts = []
        for i in range(num_layers):
            if i in layer_outputs:
                hidden = layer_outputs[i]
                last_token_hidden = hidden[0, -1, :].numpy()  # [hidden_size]
                layer_acts.append(last_token_hidden)
        
        if len(layer_acts) == num_layers:
            layer_acts = np.stack(layer_acts)  # [num_layers, hidden_size]
            all_layer_acts.append(layer_acts)
            
            # For Mistral, we'll use layer activations as head features (simplified)
            # In production, you'd extract actual attention head outputs
            head_acts = np.zeros((num_layers, num_heads, head_dim))
            for l in range(num_layers):
                # Reshape hidden states to approximate head outputs
                reshaped = layer_acts[l].reshape(num_heads, -1)[:, :head_dim]
                head_acts[l] = reshaped
            
            all_head_acts.append(head_acts.reshape(-1))
            all_labels.append(item['label'])
            all_problem_ids.append(item['problem_id'])
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert to arrays
    layer_wise = np.array(all_layer_acts)
    head_wise = np.array(all_head_acts)
    labels = np.array(all_labels)
    
    return layer_wise, head_wise, labels, all_problem_ids


def save_activations(layer_wise, head_wise, labels, problem_ids, num_layers, num_heads, split, save_dir):
    """Save extracted activations."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    np.save(save_path / f'{split}_layer_acts.npy', layer_wise)
    np.save(save_path / f'{split}_head_acts.npy', head_wise)
    np.save(save_path / f'{split}_labels.npy', labels)
    
    with open(save_path / f'{split}_problem_ids.pkl', 'wb') as f:
        pickle.dump(problem_ids, f)
    
    # Save metadata
    if split == 'train':
        metadata = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'layer_acts_shape': layer_wise.shape,
            'head_acts_shape': head_wise.shape,
            'model': MODEL_NAME
        }
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {split} activations to {save_path}")


# ================ PROBE TRAINING ================

def load_activations_from_npy(split, data_dir):
    """Load activations from numpy files."""
    data_path = Path(data_dir)
    
    layer_acts = np.load(data_path / f'{split}_layer_acts.npy')
    head_acts = np.load(data_path / f'{split}_head_acts.npy')
    labels = np.load(data_path / f'{split}_labels.npy')
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return layer_acts, head_acts, labels, metadata['num_layers'], metadata['num_heads']


def train_layer_probes(train_acts, train_labels, val_acts, val_labels):
    """Train logistic regression probes for each layer."""
    num_layers = train_acts.shape[1]
    layer_results = []
    
    print(f"\nTraining probes for {num_layers} layers...")
    
    for layer_idx in tqdm(range(num_layers), desc="Training layer probes"):
        X_train = train_acts[:, layer_idx, :]
        X_val = val_acts[:, layer_idx, :]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Try different regularization strengths
        best_val_auc = 0
        best_results = None
        
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            try:
                probe = LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs',
                    class_weight='balanced'
                )
                
                probe.fit(X_train_scaled, train_labels)
                
                train_proba = probe.predict_proba(X_train_scaled)[:, 1]
                val_proba = probe.predict_proba(X_val_scaled)[:, 1]
                
                train_acc = probe.score(X_train_scaled, train_labels)
                val_acc = probe.score(X_val_scaled, val_labels)
                
                train_auc = roc_auc_score(train_labels, train_proba)
                val_auc = roc_auc_score(val_labels, val_proba)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_results = {
                        'layer': layer_idx,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'train_auc': train_auc,
                        'val_auc': val_auc,
                        'C': C,
                        'probe': probe,
                        'scaler': scaler
                    }
            except Exception as e:
                continue
        
        if best_results is not None:
            layer_results.append(best_results)
        else:
            layer_results.append({
                'layer': layer_idx,
                'train_acc': 0.5,
                'val_acc': 0.5,
                'train_auc': 0.5,
                'val_auc': 0.5,
                'C': None,
                'probe': None,
                'scaler': None
            })
    
    return layer_results


def create_layer_importance_chart(layer_results, save_path='mistral_layer_importance_chart.png'):
    """Create visualization of layer importance for Mistral."""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extract data
    layers = [r['layer'] for r in layer_results]
    train_aucs = [r['train_auc'] for r in layer_results]
    val_aucs = [r['val_auc'] for r in layer_results]
    
    # Calculate relative importance
    importance = [max(0, auc - 0.5) for auc in val_aucs]
    if max(importance) > 0:
        normalized_importance = [i / max(importance) for i in importance]
    else:
        normalized_importance = importance
    
    # Plot 1: AUC Performance
    ax1.plot(layers, train_aucs, 'b-', label='Train AUC', alpha=0.6, linewidth=2)
    ax1.plot(layers, val_aucs, 'r-', label='Validation AUC', linewidth=2.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.fill_between(layers, 0.5, val_aucs, where=[v > 0.5 for v in val_aucs], 
                     alpha=0.2, color='green')
    
    # Highlight top 5 layers
    top_5_indices = sorted(range(len(val_aucs)), key=lambda i: val_aucs[i], reverse=True)[:5]
    for idx in top_5_indices:
        ax1.axvspan(idx - 0.3, idx + 0.3, alpha=0.15, color='green')
    
    # Highlight bottom 5 layers
    bottom_5_indices = sorted(range(len(val_aucs)), key=lambda i: val_aucs[i])[:5]
    for idx in bottom_5_indices:
        ax1.axvspan(idx - 0.3, idx + 0.3, alpha=0.15, color='red')
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('AUC Score', fontsize=12)
    ax1.set_title('Mistral-7B Layer-wise Creativity Detection Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.45, max(max(train_aucs), max(val_aucs)) + 0.05])
    
    # Plot 2: Importance Bars
    colors = plt.cm.RdYlBu_r(normalized_importance)
    bars = ax2.bar(layers, normalized_importance, color=colors, edgecolor='black', linewidth=0.5)
    
    # Highlight top 5 layers
    top_5_importance = sorted(range(len(normalized_importance)), 
                             key=lambda i: normalized_importance[i], reverse=True)[:5]
    
    for idx in top_5_importance:
        bars[idx].set_edgecolor('darkgreen')
        bars[idx].set_linewidth(2.5)
        y_pos = normalized_importance[idx] + 0.02
        ax2.text(idx, y_pos, f'L{idx}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')
    
    # Highlight bottom 5 layers
    bottom_5_importance = sorted(range(len(normalized_importance)), 
                                key=lambda i: normalized_importance[i])[:5]
    
    for idx in bottom_5_importance:
        bars[idx].set_edgecolor('darkred')
        bars[idx].set_linewidth(2.5)
        y_pos = normalized_importance[idx] + 0.02 if normalized_importance[idx] > 0.1 else 0.05
        ax2.text(idx, y_pos, f'L{idx}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Relative Importance', fontsize=12)
    ax2.set_title('Mistral-7B Relative Layer Importance for Creativity Detection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([-0.05, 1.15])
    
    # Add legend
    legend_elements = [
        Patch(facecolor='none', edgecolor='darkgreen', linewidth=2.5, label='Top 5 layers'),
        Patch(facecolor='none', edgecolor='darkred', linewidth=2.5, label='Bottom 5 layers')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.15, fraction=0.05)
    cbar.set_label('Importance Level', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")
    plt.show()
    
    return fig


# ================ MAIN PIPELINE ================

def main():
    """Main pipeline for Mistral creativity ITI."""
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Save directory: {SAVE_DIR}")
    
    # Check if we need to extract activations or can use existing ones
    if Path(SAVE_DIR).exists() and (Path(SAVE_DIR) / 'train_layer_acts.npy').exists():
        print("\n✓ Found existing activations, skipping extraction...")
    else:
        print("\n" + "="*60)
        print("STEP 1: EXTRACTING ACTIVATIONS")
        print("="*60)
        
        # Load model and tokenizer
        print(f"\nLoading Mistral model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get model dimensions
        config = model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads
        
        print(f"Model architecture: {num_layers} layers, {num_heads} heads")
        
        # Process each split
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            
            try:
                data = load_creativity_dataset(split)
                print(f"Loaded {len(data)} samples")
                
                creative_count = sum(1 for item in data if item['label'] == 1)
                print(f"  Creative: {creative_count}")
                print(f"  Non-creative: {len(data) - creative_count}")
                
                # Extract activations
                layer_wise, head_wise, labels, problem_ids = extract_activations_mistral(
                    model, tokenizer, data, DEVICE
                )
                
                print(f"\nActivations extracted successfully!")
                print(f"Layer-wise shape: {layer_wise.shape}")
                print(f"Labels shape: {labels.shape}")
                
                # Save activations
                save_activations(
                    layer_wise, head_wise, labels, problem_ids,
                    num_layers, num_heads, split, SAVE_DIR
                )
                
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print(f"Skipping {split} split...")
                continue
        
        # Clean up model from memory
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("STEP 2: TRAINING LAYER PROBES")
    print("="*60)
    
    # Load activations
    try:
        train_layer, train_head, train_labels, num_layers, num_heads = load_activations_from_npy('train', SAVE_DIR)
        val_layer, val_head, val_labels, _, _ = load_activations_from_npy('val', SAVE_DIR)
        
        print(f"\n✓ Loaded activations")
        print(f"  Train: {len(train_labels)} samples ({np.sum(train_labels)} creative)")
        print(f"  Val: {len(val_labels)} samples ({np.sum(val_labels)} creative)")
        print(f"  Model: {num_layers} layers")
        
    except Exception as e:
        print(f"Error loading activations: {e}")
        return
    
    # Train layer probes
    layer_results = train_layer_probes(train_layer, train_labels, val_layer, val_labels)
    
    # Print summary
    print("\n" + "="*60)
    print("LAYER IMPORTANCE SUMMARY")
    print("="*60)
    
    sorted_layers = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
    
    print("\nTop 5 most important layers for creativity:")
    for i, result in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. Layer {result['layer']:2d}: Val AUC = {result['val_auc']:.4f}")
    
    print("\nBottom 5 least important layers:")
    for i, result in enumerate(sorted_layers[-5:]):
        print(f"  {i+1}. Layer {result['layer']:2d}: Val AUC = {result['val_auc']:.4f}")
    
    print("\n" + "="*60)
    print("STEP 3: CREATING VISUALIZATION")
    print("="*60)
    
    create_layer_importance_chart(layer_results, save_path='mistral_layer_importance_chart.png')
    
    # Save results
    output_dir = Path('mistral_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    import csv
    with open(output_dir / 'layer_importance.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Train_AUC', 'Val_AUC', 'Train_Acc', 'Val_Acc'])
        for r in layer_results:
            writer.writerow([r['layer'], r['train_auc'], r['val_auc'], 
                           r['train_acc'], r['val_acc']])
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    print("\n" + "="*60)
    print("MISTRAL CREATIVITY ANALYSIS COMPLETE!")
    print("="*60)
    print("\n✅ Chart saved as 'mistral_layer_importance_chart.png'")
    print("📁 Results saved in 'mistral_results/' directory")


if __name__ == "__main__":
    main()