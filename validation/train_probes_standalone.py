# train_probes_standalone_fixed.py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from einops import rearrange
from tqdm import tqdm
import pickle

def train_probes_for_llama31():
    """
    Train probes on LLaMA 3.1 8B Instruct activations
    """
    print("Loading activations...")
    
    # Load activations and labels
    head_wise_activations = np.load('../features/llama3.1_8B_instruct_tqa_mc2_head_wise.npy')
    labels = np.load('../features/llama3.1_8B_instruct_tqa_mc2_labels.npy')
    
    # Model configuration for LLaMA 3.1 8B
    num_layers = 32
    num_heads = 32
    head_dim = 128
    
    print(f"Loaded activations shape: {head_wise_activations.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Reshape to separate heads
    head_wise_activations = rearrange(
        head_wise_activations, 
        'b l (h d) -> b l h d', 
        h=num_heads
    )
    print(f"Reshaped activations: {head_wise_activations.shape}")
    
    # Use raw activations without separating by question
    # since we already have individual samples
    num_samples = len(labels)
    train_size = int(0.8 * num_samples)
    
    # Shuffle indices for train/val split
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Get train and validation sets
    X_train = head_wise_activations[train_indices]
    X_val = head_wise_activations[val_indices]
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # Train probe for each head
    all_head_accs = []
    probes = []
    
    print("Training probes for each attention head...")
    for layer in tqdm(range(num_layers), desc="Layers"):
        for head in range(num_heads):
            # Get activations for this specific head
            X_train_head = X_train[:, layer, head, :]
            X_val_head = X_val[:, layer, head, :]
            
            # Train logistic regression probe
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train_head, y_train)
            
            # Evaluate on validation set
            y_val_pred = clf.predict(X_val_head)
            accuracy = accuracy_score(y_val, y_val_pred)
            
            all_head_accs.append(accuracy)
            probes.append(clf)
    
    # Convert to numpy array
    all_head_accs = np.array(all_head_accs)
    
    # Find top heads
    num_to_select = 48
    top_indices = np.argsort(all_head_accs)[-num_to_select:]
    
    top_heads = []
    for idx in top_indices:
        layer = idx // num_heads
        head = idx % num_heads
        top_heads.append((layer, head))
        
    # Sort by layer then head for readability
    top_heads = sorted(top_heads)
    
    print(f"\nTop {num_to_select} heads by probe accuracy:")
    print("="*60)
    
    # Print top heads grouped by layer
    heads_by_layer = {}
    for layer, head in top_heads:
        if layer not in heads_by_layer:
            heads_by_layer[layer] = []
        heads_by_layer[layer].append(head)
        
    for layer in sorted(heads_by_layer.keys()):
        heads = sorted(heads_by_layer[layer])
        accs = [all_head_accs[layer * num_heads + h] for h in heads]
        print(f"Layer {layer:2d}: heads {heads} (acc: {[f'{a:.3f}' for a in accs]})")
    
    # Calculate directions using center of mass
    print("\nCalculating intervention directions...")
    directions = {}
    
    # Get all training data for direction calculation
    all_train_acts = head_wise_activations[train_indices]
    all_train_labels = labels[train_indices]
    
    for layer, head in tqdm(top_heads):
        # Get activations for this head
        head_acts = all_train_acts[:, layer, head, :]
        
        # Calculate mean for true and false samples
        true_acts = head_acts[all_train_labels == 1]
        false_acts = head_acts[all_train_labels == 0]
        
        if len(true_acts) > 0 and len(false_acts) > 0:
            true_mean = np.mean(true_acts, axis=0)
            false_mean = np.mean(false_acts, axis=0)
            
            # Mass mean shift direction
            direction = true_mean - false_mean
            direction = direction / np.linalg.norm(direction)
            
            # Calculate standard deviation for scaling
            proj_vals = head_acts @ direction.T
            std = np.std(proj_vals)
            
            directions[(layer, head)] = (direction, std)
    
    # Save results
    print("\nSaving probe results...")
    results = {
        'top_heads': top_heads,
        'all_head_accs': all_head_accs,
        'probes': probes,
        'directions': directions,
        'num_layers': num_layers,
        'num_heads': num_heads
    }
    
    with open('llama31_8b_probe_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Saved probe results to llama31_8b_probe_results.pkl")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean probe accuracy: {np.mean(all_head_accs):.3f}")
    print(f"Max probe accuracy: {np.max(all_head_accs):.3f}")
    print(f"Min probe accuracy: {np.min(all_head_accs):.3f}")
    print(f"Top {num_to_select} heads mean accuracy: {np.mean(all_head_accs[top_indices]):.3f}")
    
    return top_heads, probes, all_head_accs, directions

if __name__ == "__main__":
    top_heads, probes, accuracies, directions = train_probes_for_llama31()