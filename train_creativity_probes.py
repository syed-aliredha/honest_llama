# train_creativity_probes.py
"""
Train probes to identify creativity-related attention heads.
Uses head-wise activations extracted from LLaMA model.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')


def load_activations(split='train', data_dir='features/creativity_llama31_fixed'):
    """Load saved activations from get_creativity_activations.py"""
    
    data_path = Path(data_dir)
    
    # Try loading pickle format first (contains metadata)
    pkl_path = data_path / f'{split}_activations.pkl'
    if pkl_path.exists():
        print(f"Loading {split} activations from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return (data['layer_wise'], 
                data['head_wise'], 
                data['labels'],
                data.get('num_layers', 32),
                data.get('num_heads', 32))
    
    # Fallback to numpy format
    print(f"Loading {split} activations from numpy files")
    layer_wise = np.load(data_path / f'{split}_layer_wise.npy')
    head_wise = np.load(data_path / f'{split}_head_wise.npy')
    labels = np.load(data_path / f'{split}_labels.npy')
    
    # Infer dimensions
    num_layers = layer_wise.shape[1]
    num_heads = 32  # Default for LLaMA models
    
    return layer_wise, head_wise, labels, num_layers, num_heads


def reshape_head_wise_activations(head_wise_acts, num_layers, num_heads):
    """
    Reshape head-wise activations to separate individual heads.
    
    Args:
        head_wise_acts: Shape [batch, num_layers, hidden_size]
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
    
    Returns:
        Reshaped activations: [batch, num_layers, num_heads, head_dim]
    """
    batch_size = head_wise_acts.shape[0]
    hidden_size = head_wise_acts.shape[-1]
    head_dim = hidden_size // num_heads
    
    # Reshape to separate heads
    # From [batch, layers, hidden] to [batch, layers, heads, head_dim]
    reshaped = rearrange(
        head_wise_acts, 
        'b l (h d) -> b l h d', 
        h=num_heads, 
        d=head_dim
    )
    
    return reshaped


def train_layer_probes(train_acts, train_labels, val_acts, val_labels):
    """Train a probe for each layer to identify creativity."""
    
    num_layers = train_acts.shape[1]
    
    train_scores = []
    val_scores = []
    probes = []
    
    print("\nTraining layer-wise probes...")
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        # Get activations for this layer
        X_train = train_acts[:, layer_idx, :]
        X_val = val_acts[:, layer_idx, :]
        
        # Train logistic regression probe
        probe = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42,
            solver='liblinear'  # More stable for smaller datasets
        )
        probe.fit(X_train, train_labels)
        
        # Evaluate
        train_pred = probe.predict(X_train)
        val_pred = probe.predict(X_val)
        
        train_acc = accuracy_score(train_labels, train_pred)
        val_acc = accuracy_score(val_labels, val_pred)
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        probes.append(probe)
    
    return train_scores, val_scores, probes


def train_head_probes(head_wise_acts_train, train_labels, 
                     head_wise_acts_val, val_labels,
                     num_layers, num_heads):
    """
    Train probes for each attention head.
    
    Returns:
        all_head_scores: List of dicts with head info and scores
        head_probes: List of trained probes
    """
    
    print("\nReshaping head-wise activations...")
    # Reshape to separate individual heads
    train_acts_reshaped = reshape_head_wise_activations(
        head_wise_acts_train, num_layers, num_heads
    )
    val_acts_reshaped = reshape_head_wise_activations(
        head_wise_acts_val, num_layers, num_heads
    )
    
    print(f"Reshaped train shape: {train_acts_reshaped.shape}")
    print(f"Reshaped val shape: {val_acts_reshaped.shape}")
    
    all_head_scores = []
    head_probes = []
    
    print("\nTraining head-wise probes...")
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        for head_idx in range(num_heads):
            # Get activations for this specific head
            X_train = train_acts_reshaped[:, layer_idx, head_idx, :]
            X_val = val_acts_reshaped[:, layer_idx, head_idx, :]
            
            # Train probe
            probe = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'
            )
            
            try:
                probe.fit(X_train, train_labels)
                
                # Get predictions and probabilities
                train_pred = probe.predict(X_train)
                val_pred = probe.predict(X_val)
                
                # Get probabilities for AUC
                train_proba = probe.predict_proba(X_train)[:, 1]
                val_proba = probe.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                train_acc = accuracy_score(train_labels, train_pred)
                val_acc = accuracy_score(val_labels, val_pred)
                
                # Calculate AUC if we have both classes
                if len(np.unique(train_labels)) > 1 and len(np.unique(val_labels)) > 1:
                    train_auc = roc_auc_score(train_labels, train_proba)
                    val_auc = roc_auc_score(val_labels, val_proba)
                else:
                    train_auc = val_auc = 0.5
                
            except Exception as e:
                print(f"Warning: Failed to train probe for layer {layer_idx}, head {head_idx}: {e}")
                train_acc = val_acc = train_auc = val_auc = 0.5
                probe = None
            
            head_info = {
                'layer': layer_idx,
                'head': head_idx,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_auc': train_auc,
                'auc': val_auc,  # Use validation AUC for ranking
                'probe': probe
            }
            
            all_head_scores.append(head_info)
            head_probes.append(probe)
    
    return all_head_scores, head_probes


def select_top_heads(all_head_scores, top_k=48):
    """Select top K heads based on validation AUC."""
    
    # Filter out heads with failed probes
    valid_heads = [h for h in all_head_scores if h['probe'] is not None]
    
    # Sort by AUC score
    sorted_scores = sorted(valid_heads, key=lambda x: x['auc'], reverse=True)
    
    # Get top K heads
    top_heads = sorted_scores[:min(top_k, len(sorted_scores))]
    
    print(f"\nTop {len(top_heads)} heads for creativity detection:")
    print("="*60)
    
    # Group by layer for display
    heads_by_layer = {}
    for head in top_heads:
        layer = head['layer']
        if layer not in heads_by_layer:
            heads_by_layer[layer] = []
        heads_by_layer[layer].append((head['head'], head['auc']))
    
    for layer in sorted(heads_by_layer.keys()):
        heads_info = heads_by_layer[layer]
        heads_str = ", ".join([f"H{h}({auc:.3f})" for h, auc in heads_info])
        print(f"Layer {layer:2d}: {heads_str}")
    
    return top_heads


def compute_intervention_directions(head_wise_acts, labels, top_heads, num_layers, num_heads):
    """
    Compute intervention directions using center of mass method (best performing in ITI paper).
    
    Returns:
        directions: Dict mapping (layer, head) to (direction, std)
    """
    
    print("\nComputing intervention directions using mass mean shift...")
    
    # Reshape activations to separate heads
    acts_reshaped = reshape_head_wise_activations(head_wise_acts, num_layers, num_heads)
    
    directions = {}
    
    for head_info in tqdm(top_heads, desc="Computing directions"):
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        
        # Get activations for this head
        head_acts = acts_reshaped[:, layer_idx, head_idx, :]
        
        # Separate creative and non-creative samples
        creative_acts = head_acts[labels == 1]
        non_creative_acts = head_acts[labels == 0]
        
        if len(creative_acts) > 0 and len(non_creative_acts) > 0:
            # Compute centers of mass
            creative_center = np.mean(creative_acts, axis=0)
            non_creative_center = np.mean(non_creative_acts, axis=0)
            
            # Direction from non-creative to creative (intervention direction)
            direction = creative_center - non_creative_center
            
            # Normalize direction
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                print(f"Warning: Near-zero direction for layer {layer_idx}, head {head_idx}")
                direction = np.random.randn(len(direction))
                direction = direction / np.linalg.norm(direction)
            
            # Compute standard deviation for scaling
            all_projections = head_acts @ direction.T
            std = np.std(all_projections)
            
            directions[(layer_idx, head_idx)] = (direction, std)
        else:
            print(f"Warning: Insufficient samples for layer {layer_idx}, head {head_idx}")
            # Use random direction as fallback
            dim = acts_reshaped.shape[-1]
            direction = np.random.randn(dim)
            direction = direction / np.linalg.norm(direction)
            directions[(layer_idx, head_idx)] = (direction, 1.0)
    
    return directions


def visualize_results(layer_train_scores, layer_val_scores, 
                      all_head_scores, save_dir='figures'):
    """Create visualization plots for probe performance."""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Layer-wise probe performance
    layers = list(range(len(layer_train_scores)))
    
    ax1.plot(layers, layer_train_scores, 'b-', label='Train', linewidth=2)
    ax1.plot(layers, layer_val_scores, 'r-', label='Validation', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Layer-wise Creativity Probe Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Head-wise probe performance heatmap
    # Extract valid heads only
    valid_heads = [h for h in all_head_scores if h['probe'] is not None]
    if valid_heads:
        num_layers = max([h['layer'] for h in valid_heads]) + 1
        num_heads = max([h['head'] for h in valid_heads]) + 1
        
        head_matrix = np.full((num_layers, num_heads), 0.5)  # Initialize with 0.5 (random)
        for head in valid_heads:
            head_matrix[head['layer'], head['head']] = head['auc']
        
        im = ax2.imshow(head_matrix.T, aspect='auto', cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Head')
        ax2.set_title('Head-wise Creativity Detection (AUC)')
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/creativity_probe_performance.png', dpi=150)
    plt.show()
    
    print(f"✓ Saved visualization to {save_dir}/creativity_probe_performance.png")


def save_probe_results(top_heads, directions, layer_probes, all_head_scores, 
                       num_layers, num_heads, save_dir='creativity_iti_components'):
    """Save all components needed for ITI intervention."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'top_heads': [(h['layer'], h['head']) for h in top_heads],
        'top_heads_scores': [{k: v for k, v in h.items() if k != 'probe'} for h in top_heads],
        'directions': directions,
        'layer_probes': layer_probes,
        'all_head_scores': [{k: v for k, v in h.items() if k != 'probe'} for h in all_head_scores],
        'num_layers': num_layers,
        'num_heads': num_heads
    }
    
    # Save main results
    with open(save_path / 'probe_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    # Save directions separately for easy loading
    with open(save_path / 'intervention_directions.pkl', 'wb') as f:
        pickle.dump(directions, f)
    
    print(f"✓ Saved probe results to {save_path}")
    
    return save_data


def main():
    """Main training pipeline."""
    
    print("="*60)
    print("TRAINING CREATIVITY PROBES")
    print("="*60)
    
    # Check if activation data exists
    data_dir = Path('features/creativity_llama31_fixed')
    if not data_dir.exists():
        print(f"Error: Activation data not found in {data_dir}")
        print("Please run get_creativity_activations.py first.")
        return
    
    # Load activations
    print("\nLoading activations...")
    try:
        train_layer, train_head, train_labels, num_layers, num_heads = load_activations('train')
        val_layer, val_head, val_labels, _, _ = load_activations('val')
        test_layer, test_head, test_labels, _, _ = load_activations('test')
    except Exception as e:
        print(f"Error loading activations: {e}")
        return
    
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_labels)} samples ({np.sum(train_labels)} creative)")
    print(f"  Val: {len(val_labels)} samples ({np.sum(val_labels)} creative)")
    print(f"  Test: {len(test_labels)} samples ({np.sum(test_labels)} creative)")
    print(f"\nModel architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads per layer: {num_heads}")
    
    # Train layer-wise probes
    print("\n" + "="*60)
    print("LAYER-WISE PROBE TRAINING")
    print("="*60)
    
    layer_train_scores, layer_val_scores, layer_probes = train_layer_probes(
        train_layer, train_labels, val_layer, val_labels
    )
    
    best_layer = np.argmax(layer_val_scores)
    print(f"\nBest layer: {best_layer} (val acc: {layer_val_scores[best_layer]:.3f})")
    
    # Train head-wise probes
    print("\n" + "="*60)
    print("HEAD-WISE PROBE TRAINING")
    print("="*60)
    
    all_head_scores, head_probes = train_head_probes(
        train_head, train_labels,
        val_head, val_labels,
        num_layers, num_heads
    )
    
    # Select top heads
    top_k = 48  # Standard for ITI
    top_heads = select_top_heads(all_head_scores, top_k=top_k)
    
    # Compute intervention directions
    print("\n" + "="*60)
    print("COMPUTING INTERVENTION DIRECTIONS")
    print("="*60)
    
    # Combine train and validation for direction computation (following ITI paper)
    combined_head_acts = np.concatenate([train_head, val_head], axis=0)
    combined_labels = np.concatenate([train_labels, val_labels], axis=0)
    
    directions = compute_intervention_directions(
        combined_head_acts, combined_labels,
        top_heads, num_layers, num_heads
    )
    
    print(f"✓ Computed directions for {len(directions)} heads")
    
    # Test set evaluation
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Evaluate best layer probe on test set
    best_probe = layer_probes[best_layer]
    test_pred = best_probe.predict(test_layer[:, best_layer, :])
    test_acc = accuracy_score(test_labels, test_pred)
    
    print(f"Test accuracy (best layer {best_layer}): {test_acc:.3f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, 
                              target_names=['Non-creative', 'Creative']))
    
    # Visualize results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    visualize_results(layer_train_scores, layer_val_scores, all_head_scores)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    save_data = save_probe_results(
        top_heads, directions, layer_probes, all_head_scores,
        num_layers, num_heads
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  ✓ Trained probes for {num_layers} layers")
    print(f"  ✓ Identified top {len(top_heads)} heads for creativity")
    print(f"  ✓ Computed intervention directions")
    print(f"  ✓ Test accuracy: {test_acc:.3f}")
    print(f"\nNext steps:")
    print("  1. Use apply_creativity_iti.py to apply interventions")
    print("  2. Evaluate on creativity benchmarks")


if __name__ == "__main__":
    main()