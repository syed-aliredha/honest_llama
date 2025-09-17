"""
Layer-wise Importance Analysis for Creativity Detection
Works with your actual data format (separate .npy files)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')


def load_activations_from_npy(split='train', data_dir='features/creativity_llama31_fixed'):
    """Load activations from separate .npy files (your actual format)."""
    data_path = Path(data_dir)
    
    print(f"Loading {split} activations from {data_path}...")
    
    try:
        # Load individual numpy files
        layer_wise = np.load(data_path / f'{split}_layer_wise.npy')
        head_wise = np.load(data_path / f'{split}_head_wise.npy')
        labels = np.load(data_path / f'{split}_labels.npy')
        
        # Try to load the pickle file for additional info
        try:
            with open(data_path / f'{split}_activations.pkl', 'rb') as f:
                pkl_data = pickle.load(f)
                # Extract any additional info if needed
        except:
            pass
        
        # Infer architecture from shapes
        num_layers = layer_wise.shape[1]
        # Assuming head_wise shape is (n_samples, n_layers * n_heads * head_dim)
        # or (n_samples, total_features)
        # We'll calculate approximate n_heads
        if head_wise.ndim == 2:
            total_head_features = head_wise.shape[1]
            head_dim = layer_wise.shape[2] // 32  # Common head count
            num_heads = 32  # Default assumption for LLaMA
        else:
            num_heads = 32  # Default
        
        print(f"  ✓ Loaded {len(labels)} samples")
        print(f"  ✓ Layer-wise shape: {layer_wise.shape}")
        print(f"  ✓ Head-wise shape: {head_wise.shape}")
        print(f"  ✓ Labels shape: {labels.shape}")
        
        return layer_wise, head_wise, labels, num_layers, num_heads
        
    except Exception as e:
        print(f"Error loading {split} data: {e}")
        raise


def train_layer_probes(train_acts, train_labels, val_acts, val_labels):
    """Train a probe for each layer."""
    num_layers = train_acts.shape[1]
    layer_results = []
    
    print(f"\nTraining probes for {num_layers} layers...")
    
    for layer_idx in tqdm(range(num_layers), desc="Training layer probes"):
        # Extract this layer's activations
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
                # Train probe
                probe = LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs',
                    class_weight='balanced'
                )
                
                probe.fit(X_train_scaled, train_labels)
                
                # Get predictions and probabilities
                train_proba = probe.predict_proba(X_train_scaled)[:, 1]
                val_proba = probe.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate metrics
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
            # If all attempts failed, add placeholder
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


def create_layer_importance_chart(layer_results, save_path='layer_importance_chart.png'):
    """Create a clean visualization of layer importance."""
    
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
    
    # Highlight top 5 performing layers (best)
    top_5_indices = sorted(range(len(val_aucs)), key=lambda i: val_aucs[i], reverse=True)[:5]
    for idx in top_5_indices:
        ax1.axvspan(idx - 0.3, idx + 0.3, alpha=0.15, color='green')
    
    # Highlight bottom 5 performing layers (worst)
    bottom_5_indices = sorted(range(len(val_aucs)), key=lambda i: val_aucs[i])[:5]
    for idx in bottom_5_indices:
        ax1.axvspan(idx - 0.3, idx + 0.3, alpha=0.15, color='red')
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('AUC Score', fontsize=12)
    ax1.set_title('Layer-wise Creativity Detection Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.45, max(max(train_aucs), max(val_aucs)) + 0.05])
    
    # Plot 2: Importance Bars
    colors = plt.cm.RdYlBu_r(normalized_importance)
    bars = ax2.bar(layers, normalized_importance, color=colors, edgecolor='black', linewidth=0.5)
    
    # Highlight top 5 layers (best - thick green border)
    top_5_importance = sorted(range(len(normalized_importance)), 
                             key=lambda i: normalized_importance[i], reverse=True)[:5]
    
    for idx in top_5_importance:
        bars[idx].set_edgecolor('darkgreen')
        bars[idx].set_linewidth(2.5)
        ax2.text(idx, normalized_importance[idx] + 0.02, f'L{idx}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
    
    # Highlight bottom 5 layers (worst - thick red border)
    bottom_5_importance = sorted(range(len(normalized_importance)), 
                                key=lambda i: normalized_importance[i])[:5]
    
    for idx in bottom_5_importance:
        bars[idx].set_edgecolor('darkred')
        bars[idx].set_linewidth(2.5)
        # Add labels for bottom layers (positioned slightly above the bar)
        y_pos = normalized_importance[idx] + 0.02 if normalized_importance[idx] > 0.1 else 0.05
        ax2.text(idx, y_pos, f'L{idx}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Relative Importance', fontsize=12)
    ax2.set_title('Relative Layer Importance for Creativity Detection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([-0.05, 1.15])
    
    # Add legend for the highlighting
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


def print_summary(layer_results):
    """Print summary of results."""
    # Sort by validation AUC
    sorted_results = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT LAYERS")
    print("="*60)
    print(f"{'Rank':<6} {'Layer':<8} {'Val AUC':<12} {'Train AUC':<12} {'Overfit':<10}")
    print("-"*60)
    
    for i, result in enumerate(sorted_results[:10], 1):
        overfit = result['train_auc'] - result['val_auc']
        print(f"{i:<6} {result['layer']:<8} {result['val_auc']:<12.4f} "
              f"{result['train_auc']:<12.4f} {overfit:<10.4f}")
    
    # Show bottom 5 layers
    print("\n" + "="*60)
    print("BOTTOM 5 LEAST IMPORTANT LAYERS")
    print("="*60)
    print(f"{'Rank':<6} {'Layer':<8} {'Val AUC':<12} {'Train AUC':<12} {'Overfit':<10}")
    print("-"*60)
    
    bottom_5 = sorted_results[-5:]
    for i, result in enumerate(reversed(bottom_5), 1):
        overfit = result['train_auc'] - result['val_auc']
        print(f"{i:<6} {result['layer']:<8} {result['val_auc']:<12.4f} "
              f"{result['train_auc']:<12.4f} {overfit:<10.4f}")
    
    # Calculate group statistics
    num_layers = len(layer_results)
    early_layers = layer_results[:num_layers//3]
    middle_layers = layer_results[num_layers//3:2*num_layers//3]
    late_layers = layer_results[2*num_layers//3:]
    
    print("\n" + "="*60)
    print("LAYER GROUP ANALYSIS")
    print("="*60)
    
    for group_name, group in [("Early", early_layers), ("Middle", middle_layers), ("Late", late_layers)]:
        avg_val_auc = np.mean([r['val_auc'] for r in group])
        max_val_auc = max([r['val_auc'] for r in group])
        min_val_auc = min([r['val_auc'] for r in group])
        best_layer = max(group, key=lambda x: x['val_auc'])['layer']
        worst_layer = min(group, key=lambda x: x['val_auc'])['layer']
        
        print(f"\n{group_name} Layers (Layers {group[0]['layer']}-{group[-1]['layer']}):")
        print(f"  Average Val AUC: {avg_val_auc:.4f}")
        print(f"  Best Val AUC: {max_val_auc:.4f} (Layer {best_layer})")
        print(f"  Worst Val AUC: {min_val_auc:.4f} (Layer {worst_layer})")


def test_best_layers(layer_results, test_acts, test_labels):
    """Test the best layers on test set."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Get top 5 layers
    sorted_results = sorted(layer_results, key=lambda x: x['val_auc'], reverse=True)[:5]
    
    print("\nTop 5 layers on test set:")
    print("-"*50)
    
    for i, result in enumerate(sorted_results, 1):
        if result['probe'] is not None:
            layer_idx = result['layer']
            X_test = test_acts[:, layer_idx, :]
            X_test_scaled = result['scaler'].transform(X_test)
            
            test_proba = result['probe'].predict_proba(X_test_scaled)[:, 1]
            test_auc = roc_auc_score(test_labels, test_proba)
            test_acc = result['probe'].score(X_test_scaled, test_labels)
            
            print(f"{i}. Layer {layer_idx:2d}: Test AUC={test_auc:.4f}, "
                  f"Test Acc={test_acc:.4f} (Val AUC={result['val_auc']:.4f})")


def save_results(layer_results, output_dir='layer_analysis_results'):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as CSV
    import csv
    with open(output_path / 'layer_importance.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Train_AUC', 'Val_AUC', 'Train_Acc', 'Val_Acc'])
        for r in layer_results:
            writer.writerow([r['layer'], r['train_auc'], r['val_auc'], 
                           r['train_acc'], r['val_acc']])
    
    print(f"\n✓ Results saved to {output_path}/")


def main():
    """Main function."""
    print("="*60)
    print("LAYER-WISE CREATIVITY IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Configuration
    data_dir = 'features/creativity_llama31_fixed'
    
    # Check if data exists
    if not Path(data_dir).exists():
        print(f"\nError: Data directory '{data_dir}' not found!")
        return
    
    # Load data using the correct format (separate .npy files)
    try:
        train_layer, train_head, train_labels, num_layers, num_heads = load_activations_from_npy('train', data_dir)
        val_layer, val_head, val_labels, _, _ = load_activations_from_npy('val', data_dir)
        
        print(f"\n✓ Successfully loaded data")
        print(f"  Model: {num_layers} layers")
        print(f"  Train: {len(train_labels)} samples ({np.sum(train_labels)} creative)")
        print(f"  Val: {len(val_labels)} samples ({np.sum(val_labels)} creative)")
        
        # Try to load test data
        try:
            test_layer, test_head, test_labels, _, _ = load_activations_from_npy('test', data_dir)
            print(f"  Test: {len(test_labels)} samples ({np.sum(test_labels)} creative)")
            has_test = True
        except:
            print("  Test: Not available")
            has_test = False
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train layer probes
    print("\n" + "="*60)
    print("TRAINING LAYER-WISE PROBES")
    print("="*60)
    
    layer_results = train_layer_probes(train_layer, train_labels, val_layer, val_labels)
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    create_layer_importance_chart(layer_results)
    
    # Print summary
    print_summary(layer_results)
    
    # Test on test set if available
    if has_test:
        test_best_layers(layer_results, test_layer, test_labels)
    
    # Save results
    save_results(layer_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\n✅ Chart saved as 'layer_importance_chart.png'")
    print("📁 Results saved in 'layer_analysis_results/' directory")


if __name__ == "__main__":
    main()