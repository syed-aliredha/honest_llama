# create_creativity_dataset_with_solutions.py
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

def analyze_creativity_data():
    """Analyze the creativity dataset and prepare for ITI"""
    
    # Load the CSV
    df = pd.read_csv('Llama-8B-Instruct_dola3_creativity.csv')
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Total problems: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    # Analyze correctness
    correct_count = df['correctness'].sum()
    print(f"Correct solutions: {correct_count} ({correct_count/len(df)*100:.1f}%)")
    
    # Analyze creativity metrics
    has_new_techniques = df['new_techniques'] > 0
    creative_count = (df['correctness'] & has_new_techniques).sum()
    print(f"Creative solutions (correct + new techniques): {creative_count} ({creative_count/len(df)*100:.1f}%)")
    
    # Distribution of new_techniques_ratio
    print(f"\nNew techniques ratio stats:")
    print(f"  Mean: {df['new_techniques_ratio'].mean():.3f}")
    print(f"  Std: {df['new_techniques_ratio'].std():.3f}")
    print(f"  Max: {df['new_techniques_ratio'].max():.3f}")
    print(f"  Non-zero count: {(df['new_techniques_ratio'] > 0).sum()}")
    
    # Check follow_constraints
    follow_constraints_count = df['follow_constraints'].sum()
    print(f"\nFollow constraints: {follow_constraints_count} ({follow_constraints_count/len(df)*100:.1f}%)")
    
    return df

def prepare_balanced_dataset(df, creativity_threshold=0.1, max_samples_per_class=50):
    """Create a balanced dataset for probe training"""
    
    # Define creative samples: correct AND has new techniques above threshold
    df['is_creative'] = (df['correctness'] == True) & (df['new_techniques_ratio'] >= creativity_threshold)
    
    # Define non-creative samples: correct but NO new techniques
    df['is_non_creative'] = (df['correctness'] == True) & (df['new_techniques_ratio'] == 0)
    
    creative_samples = df[df['is_creative']]
    non_creative_samples = df[df['is_non_creative']]
    
    print("\n" + "=" * 60)
    print("BALANCED DATASET CREATION")
    print("=" * 60)
    print(f"Creative samples available: {len(creative_samples)}")
    print(f"Non-creative samples available: {len(non_creative_samples)}")
    
    # Balance the dataset
    n_creative = min(len(creative_samples), max_samples_per_class)
    n_non_creative = min(len(non_creative_samples), max_samples_per_class)
    n_samples = min(n_creative, n_non_creative)
    
    balanced_creative = creative_samples.sample(n=n_samples, random_state=42)
    balanced_non_creative = non_creative_samples.sample(n=n_samples, random_state=42)
    
    balanced_df = pd.concat([balanced_creative, balanced_non_creative]).reset_index(drop=True)
    balanced_df['label'] = balanced_df['is_creative'].astype(int)
    
    print(f"Balanced dataset size: {len(balanced_df)} ({n_samples} per class)")
    
    return balanced_df

def create_prompts_with_full_solutions(balanced_df):
    """Convert dataset to prompts including full solutions minus last token"""
    
    prompts_data = []
    
    for idx, row in balanced_df.iterrows():
        # Extract problem constraints
        constraints = eval(row['constraints']) if isinstance(row['constraints'], str) else row['constraints']
        
        # Get the solution and remove last token
        full_solution = row['machine_solutions'].strip()
        
        # More sophisticated last token removal
        # Remove the last meaningful token (not just whitespace)
        lines = full_solution.split('\n')
        last_line = lines[-1] if lines else ''
        
        if last_line.strip():
            # If last line has content, truncate it
            tokens = last_line.split()
            if tokens:
                # Remove last token from last line
                truncated_last_line = ' '.join(tokens[:-1])
                lines[-1] = truncated_last_line
                partial_solution = '\n'.join(lines)
            else:
                # If no tokens, remove the line
                partial_solution = '\n'.join(lines[:-1])
        else:
            # If last line is empty, look for previous line with content
            partial_solution = full_solution.rstrip()
            # Remove last character if it's meaningful
            if partial_solution and partial_solution[-1] not in '\n\r\t ':
                partial_solution = partial_solution[:-1]
        
        # Create prompt with almost-complete solution
        prompt = f"""Problem {row['problem_id']}:
Write a Python function to solve this problem with the following constraints:
{', '.join(constraints)}

Here is a solution that needs completion:
```python
{partial_solution}"""
        
        prompts_data.append({
            'problem_id': row['problem_id'],
            'prompt': prompt,
            'full_solution': row['machine_solutions'],
            'partial_solution': partial_solution,
            'is_creative': row['is_creative'],
            'label': row['label'],
            'new_techniques_ratio': row['new_techniques_ratio'],
            'constraints': constraints,
            'human_techniques': eval(row['human_techniques']) if isinstance(row['human_techniques'], str) else row['human_techniques'],
            'machine_techniques': eval(row['machine_techniques']) if isinstance(row['machine_techniques'], str) else row['machine_techniques']
        })
    
    return prompts_data

def save_datasets(prompts_data, test_size=0.2, val_size=0.2):
    """Split and save the dataset for ITI training"""
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        prompts_data, test_size=test_size, 
        stratify=[d['label'] for d in prompts_data],
        random_state=42
    )
    
    # Second split: train vs val
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size/(1-test_size),
        stratify=[d['label'] for d in train_val_data],
        random_state=42
    )
    
    print("\n" + "=" * 60)
    print("DATASET SPLITS")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"  Creative: {sum(d['label'] for d in train_data)}")
    print(f"  Non-creative: {sum(1-d['label'] for d in train_data)}")
    
    print(f"\nValidation samples: {len(val_data)}")
    print(f"  Creative: {sum(d['label'] for d in val_data)}")
    print(f"  Non-creative: {sum(1-d['label'] for d in val_data)}")
    
    print(f"\nTest samples: {len(test_data)}")
    print(f"  Creative: {sum(d['label'] for d in test_data)}")
    print(f"  Non-creative: {sum(1-d['label'] for d in test_data)}")
    
    # Save datasets
    Path('creativity_data').mkdir(exist_ok=True)
    
    with open('creativity_data/train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('creativity_data/val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open('creativity_data/test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # Also save as JSON for readability
    with open('creativity_data/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print("\n✓ Datasets saved to creativity_data/")
    
    return train_data, val_data, test_data

def main():
    # Analyze the data
    df = analyze_creativity_data()
    
    # Create balanced dataset
    balanced_df = prepare_balanced_dataset(df, creativity_threshold=0.1, max_samples_per_class=40)
    
    # Convert to prompts WITH solutions
    prompts_data = create_prompts_with_full_solutions(balanced_df)
    
    # Save splits
    train_data, val_data, test_data = save_datasets(prompts_data)
    
    # Show example
    print("\n" + "=" * 60)
    print("EXAMPLE PROMPT WITH SOLUTION")
    print("=" * 60)
    creative_example = [d for d in train_data if d['label'] == 1][0]
    print("CREATIVE EXAMPLE:")
    print(creative_example['prompt'])
    print("\nOriginal full solution:")
    print(creative_example['full_solution'])
    print(f"\nNew techniques ratio: {creative_example['new_techniques_ratio']:.2f}")
    print(f"Human techniques: {creative_example['human_techniques']}")
    print(f"Machine techniques: {creative_example['machine_techniques']}")
    
    print("\n" + "-" * 60)
    
    non_creative_example = [d for d in train_data if d['label'] == 0][0]
    print("NON-CREATIVE EXAMPLE:")
    print(non_creative_example['prompt'])
    print(f"\nNew techniques ratio: {non_creative_example['new_techniques_ratio']:.2f}")

if __name__ == "__main__":
    main()