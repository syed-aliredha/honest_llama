# train_creativity_probes.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_activations(split='train'):
    """Load saved activations"""
    with open(f'features/creativity_llama31/{split}_activations.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['layer_wise'], data['labels']

def train_probe_per_layer(train_acts, train_labels, val_acts, val_labels):
    """Train a probe for each layer and return accuracies"""
    num_layers = train_acts.shape[1]
    
    train_scores = []
    val_scores = []
    probes = []
    
    for layer_idx in tqdm(range(num_layers), desc="Training layer probes"):
        # Get activations for this layer
        X_train = train_acts[:, layer_idx, :]
        X_val = val_acts[:, layer_idx, :]
        
        # Train logistic regression probe
        probe = LogisticRegression(max_iter=1000, class_weight='balanced')
        probe.fit(X_train, train_labels)
        
        # Evaluate
        train_pred = probe.predict(X_train)
        val_pred = probe.predict(X_val)
        
        train_acc = accuracy_score(train_labels, train_pred)
        val_acc = accuracy_score(val_labels, val_pred)
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        probes.append(probe)
        
        print(f"Layer {layer_idx}: Train acc={train_acc:.3f}, Val acc={val_acc:.3f}")
    
    return train_scores, val_scores, probes

def simulate_head_probes(layer_acts, num_heads=32):
    """
    Simulate head-wise probes by splitting layer activations into chunks
    This is a simplified approach since we don't have true head-wise activations
    """
    num_samples, num_layers, hidden_dim = layer_acts.shape
    head_dim = hidden_dim // num_heads
    
    # Reshape to simulate heads
    head_acts = layer_acts.reshape(num_samples, num_layers, num_heads, head_dim)
    
    return head_acts

def find_top_heads(train_acts, train_labels, val_acts, val_labels, num_heads=32, top_k=48):
    """Find the top K heads that best predict creativity"""
    
    # Simulate head-wise activations
    train_head_acts = simulate_head_probes(train_acts, num_heads)
    val_head_acts = simulate_head_probes(val_acts, num_heads)
    
    num_layers = train_head_acts.shape[1]
    
    head_scores = []
    
    print(f"\nTraining probes for {num_layers} layers × {num_heads} heads...")
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        for head_idx in range(num_heads):
            # Get activations for this head
            X_train = train_head_acts[:, layer_idx, head_idx, :]
            X_val = val_head_acts[:, layer_idx, head_idx, :]
            
            # Train probe
            probe = LogisticRegression(max_iter=500, class_weight='balanced')
            try:
                probe.fit(X_train, train_labels)
                
                # Get validation accuracy
                val_pred = probe.predict(X_val)
                val_acc = accuracy_score(val_labels, val_pred)
                
                # Also get probability scores for AUC
                if hasattr(probe, 'predict_proba'):
                    val_proba = probe.predict_proba(X_val)[:, 1]
                    val_auc = roc_auc_score(val_labels, val_proba)
                else:
                    val_auc = val_acc
                
                head_scores.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'accuracy': val_acc,
                    'auc': val_auc,
                    'probe': probe
                })
            except:
                # Some heads might not converge
                head_scores.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'accuracy': 0.5,
                    'auc': 0.5,
                    'probe': None
                })
    
    # Sort by AUC score
    head_scores.sort(key=lambda x: x['auc'], reverse=True)
    
    # Get top K heads
    top_heads = head_scores[:top_k]
    
    print(f"\nTop {top_k} heads for creativity detection:")
    for i, head in enumerate(top_heads[:10]):  # Show top 10
        print(f"  {i+1}. Layer {head['layer']}, Head {head['head']}: AUC={head['auc']:.3f}")
    
    return top_heads, head_scores

def compute_creativity_directions(train_acts, train_labels, top_heads, num_heads=32):
    """Compute intervention directions using center of mass"""
    
    # Simulate head-wise activations
    train_head_acts = simulate_head_probes(train_acts, num_heads)
    
    directions = {}
    
    for head_info in top_heads:
        layer_idx = head_info['layer']
        head_idx = head_info['head']
        
        # Get activations for this head
        head_acts = train_head_acts[:, layer_idx, head_idx, :]
        
        # Separate creative and non-creative samples
        creative_acts = head_acts[train_labels == 1]
        non_creative_acts = head_acts[train_labels == 0]
        
        # Compute centers of mass
        creative_center = creative_acts.mean(axis=0)
        non_creative_center = non_creative_acts.mean(axis=0)
        
        # Direction from non-creative to creative
        direction = creative_center - non_creative_center
        
        # Normalize
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        directions[(layer_idx, head_idx)] = direction
    
    return directions

def visualize_probe_performance(train_scores, val_scores):
    """Visualize probe performance across layers"""
    
    plt.figure(figsize=(12, 6))
    
    layers = list(range(len(train_scores)))
    
    plt.subplot(1, 2, 1)
    plt.plot(layers, train_scores, 'b-', label='Train')
    plt.plot(layers, val_scores, 'r-', label='Validation')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title('Creativity Probe Performance by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(layers, val_scores, color='skyblue', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Layer')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('creativity_probe_performance.png', dpi=150)
    plt.show()
    print("✓ Saved probe performance plot to creativity_probe_performance.png")

def save_iti_components(top_heads, directions, probes):
    """Save all components needed for ITI"""
    
    save_dir = Path('creativity_iti_components')
    save_dir.mkdir(exist_ok=True)
    
    # Save top heads info
    with open(save_dir / 'top_heads.pkl', 'wb') as f:
        # Remove probe objects for serialization
        heads_to_save = [{k: v for k, v in h.items() if k != 'probe'} 
                         for h in top_heads]
        pickle.dump(heads_to_save, f)
    
    # Save directions
    with open(save_dir / 'directions.pkl', 'wb') as f:
        pickle.dump(directions, f)
    
    # Save layer-wise probes
    with open(save_dir / 'layer_probes.pkl', 'wb') as f:
        pickle.dump(probes, f)
    
    print(f"✓ Saved ITI components to {save_dir}/")

def main():
    print("=" * 60)
    print("TRAINING CREATIVITY PROBES")
    print("=" * 60)
    
    # Load activations
    train_acts, train_labels = load_activations('train')
    val_acts, val_labels = load_activations('val')
    test_acts, test_labels = load_activations('test')
    
    print(f"Loaded activations:")
    print(f"  Train: {train_acts.shape}, {np.sum(train_labels)} creative")
    print(f"  Val: {val_acts.shape}, {np.sum(val_labels)} creative")
    print(f"  Test: {test_acts.shape}, {np.sum(test_labels)} creative")
    
    # Train layer-wise probes
    print("\n" + "=" * 60)
    print("LAYER-WISE PROBES")
    print("=" * 60)
    train_scores, val_scores, layer_probes = train_probe_per_layer(
        train_acts, train_labels, val_acts, val_labels
    )
    
    # Find best layers
    best_layer = np.argmax(val_scores)
    print(f"\nBest layer: {best_layer} with validation accuracy {val_scores[best_layer]:.3f}")
    
    # Visualize
    visualize_probe_performance(train_scores, val_scores)
    
    # Find top heads
    print("\n" + "=" * 60)
    print("HEAD-WISE ANALYSIS")
    print("=" * 60)
    top_heads, all_head_scores = find_top_heads(
        train_acts, train_labels, val_acts, val_labels,
        num_heads=32, top_k=48
    )
    
    # Compute intervention directions
    print("\n" + "=" * 60)
    print("COMPUTING INTERVENTION DIRECTIONS")
    print("=" * 60)
    directions = compute_creativity_directions(
        train_acts, train_labels, top_heads, num_heads=32
    )
    print(f"✓ Computed directions for {len(directions)} heads")
    
    # Test on held-out set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    # Use best layer probe
    best_probe = layer_probes[best_layer]
    test_pred = best_probe.predict(test_acts[:, best_layer, :])
    test_acc = accuracy_score(test_labels, test_pred)
    
    print(f"Test accuracy (best layer): {test_acc:.3f}")
    
    # Save everything
    save_iti_components(top_heads, directions, layer_probes)
    
    print("\n✓ Probe training complete!")
    print(f"Ready for ITI with {len(top_heads)} selected heads")

if __name__ == "__main__":
    main()