#!/usr/bin/env python3
"""
Simple script to export Layer and Relative Importance to CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

def export_importance(model_name):
    """Export layer and relative importance for a model."""
    
    # Determine paths based on model
    if model_name.lower() == 'mistral':
        input_path = 'mistral_results/layer_importance.csv'
        output_path = 'mistral_plot.csv'
    elif model_name.lower() == 'gemma':
        input_path = 'gemma_results/layer_importance.csv'
        output_path = 'gemma_plot.csv'
    else:
        print(f"Unknown model: {model_name}")
        return
    
    # Check if file exists
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found. Run the analysis first.")
        return
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Calculate relative importance
    val_aucs = df['Val_AUC'].values
    importance = np.maximum(0, val_aucs - 0.5)  # Above random baseline
    
    # Normalize to 0-1
    if importance.max() > 0:
        relative_importance = importance / importance.max()
    else:
        relative_importance = importance
    
    # Create output
    output_df = pd.DataFrame({
        'Layer': df['Layer'].values,
        'Relative_Importance': relative_importance
    })
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved {output_path}")
    print(f"   Layers: {len(output_df)}")
    print(f"   Max importance: {relative_importance.max():.4f} (Layer {df['Layer'].values[relative_importance.argmax()]})")
    print(f"   Min importance: {relative_importance.min():.4f} (Layer {df['Layer'].values[relative_importance.argmin()]})")


# Run for both models
if __name__ == "__main__":
    print("Exporting layer importance data...\n")
    export_importance('mistral')
    export_importance('gemma')
    print("\nDone! Check mistral_plot.csv and gemma_plot.csv")