import pandas as pd
import numpy as np
import json
import pickle
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

percentage=0.6
percentage_name='60%'

def analyze_creativity_data():
    """Load and analyze the creativity dataset."""
    
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
    """Create a balanced dataset for probe training."""
    
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


def count_code_characters(text):
    """Count non-whitespace, non-tab characters in text."""
    # Remove all whitespace characters (spaces, tabs, newlines) for counting
    cleaned = re.sub(r'[\s\t]', '', text)
    return len(cleaned)


def extract_partial_solution(solution, percentage=percentage):
    """
    Extract partial solution based on character count (excluding whitespace/tabs).
    
    Args:
        solution: Full solution string
        percentage: Target percentage of characters to include (default 0.3)
    
    Returns:
        Partial solution string
    """
    if not solution or not solution.strip():
        return ""
    
    lines = solution.strip().split('\n')
    total_lines = len(lines)
    
    # Count total code characters (excluding whitespace/tabs)
    total_chars = count_code_characters(solution)
    target_chars = int(total_chars * percentage)
    
    # If very short solution, return at least first line
    if total_lines <= 1 or total_chars <= 50:
        return lines[0] if lines else ""
    
    # Good ending patterns for clean cuts
    good_endings = [
        r'^\s*return\s+',  # return statement
        r'^\s*$',  # empty line
        r'^\s*#',  # comment
        r'^\s*\w+\s*=',  # assignment
        r'^\s*print\(',  # print statement
        r'^\s*if\s+.*:$',  # if statement start
        r'^\s*for\s+.*:$',  # for loop start
        r'^\s*while\s+.*:$',  # while loop start
        r'^\s*def\s+.*:$',  # function definition
        r'^\s*class\s+.*:$',  # class definition
        r'^\s*try:$',  # try block
        r'^\s*except.*:$',  # except block
        r'^\s*else:$',  # else block
        r'^\s*elif\s+.*:$',  # elif statement
        r'^\s*pass$',  # pass statement
        r'^\s*break$',  # break statement
        r'^\s*continue$',  # continue statement
    ]
    
    # Build up solution line by line until we reach target characters
    current_chars = 0
    best_cutoff = 1
    reached_target = False
    
    for i, line in enumerate(lines):
        line_chars = count_code_characters(line)
        current_chars += line_chars
        
        # If we've reached at least our target
        if current_chars >= target_chars and not reached_target:
            reached_target = True
            best_cutoff = i + 1
            
            # Look ahead for a good stopping point (within next few lines)
            search_end = min(i + 5, total_lines)
            
            for j in range(i, search_end):
                if j >= len(lines):
                    break
                    
                check_line = lines[j]
                
                # Check if this line is a good ending point
                for pattern in good_endings:
                    if re.match(pattern, check_line):
                        best_cutoff = j + 1
                        break
                
                # Check indentation - if next line has less indentation, current is complete
                if j + 1 < len(lines):
                    next_line = lines[j + 1]
                    current_indent = len(check_line) - len(check_line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    if next_indent <= current_indent and next_line.strip():
                        best_cutoff = j + 1
                        break
                
                # Update character count
                if j > i:
                    current_chars += count_code_characters(lines[j])
                
                # Don't go too far over target (max 110% of target)
                if current_chars > target_chars * 1.1:
                    break
            
            break
    
    # If we never reached target, include all lines
    if not reached_target:
        best_cutoff = total_lines
    
    partial_solution = '\n'.join(lines[:best_cutoff])
    
    # Clean up incomplete syntax at the end
    if partial_solution.rstrip().endswith((':', ',', '\\', '(', '[', '{')):
        # Try to include one more line if possible
        if best_cutoff < len(lines):
            partial_solution = '\n'.join(lines[:best_cutoff + 1])
        # Or remove the incomplete line
        elif best_cutoff > 1:
            # Check if removing last line still gives us reasonable coverage
            temp_solution = '\n'.join(lines[:best_cutoff - 1])
            temp_chars = count_code_characters(temp_solution)
            # Only remove if we still have at least 70% of target
            if temp_chars >= target_chars * 0.7:
                partial_solution = temp_solution
    
    return partial_solution


def create_prompts_with_partial_solutions(balanced_df, percentage=percentage):
    """Convert dataset to prompts including partial solutions based on character count."""
    
    prompts_data = []
    
    for idx, row in balanced_df.iterrows():
        # Extract problem constraints
        constraints = eval(row['constraints']) if isinstance(row['constraints'], str) else row['constraints']
        
        # Get partial solution (30% by character count)
        full_solution = row['machine_solutions'].strip()
        partial_solution = extract_partial_solution(full_solution, percentage)
        
        # Calculate actual percentage based on characters
        full_chars = count_code_characters(full_solution)
        partial_chars = count_code_characters(partial_solution)
        actual_percentage = partial_chars / max(full_chars, 1)
        
        # Also calculate line percentage for comparison
        full_lines = len(full_solution.split('\n'))
        partial_lines = len(partial_solution.split('\n'))
        line_percentage = partial_lines / max(full_lines, 1)
        
        # Create prompt with partial solution
        prompt = f"""Problem {row['problem_id']}:
Write a Python function to solve this problem with the following constraints:
{', '.join(constraints)}

Here is a partial solution:
```python
{partial_solution}
```

Complete this solution:"""
        
        prompts_data.append({
            'problem_id': row['problem_id'],
            'prompt': prompt,
            'full_solution': row['machine_solutions'],
            'partial_solution': partial_solution,
            'partial_percentage': actual_percentage,  # Character-based percentage (primary)
            'partial_line_percentage': line_percentage,  # Line-based percentage (for reference)
            'full_char_count': full_chars,
            'partial_char_count': partial_chars,
            'is_creative': row['is_creative'],
            'label': row['label'],
            'new_techniques_ratio': row['new_techniques_ratio'],
            'constraints': constraints,
            'human_techniques': eval(row['human_techniques']) if isinstance(row['human_techniques'], str) else row['human_techniques'],
            'machine_techniques': eval(row['machine_techniques']) if isinstance(row['machine_techniques'], str) else row['machine_techniques']
        })
    
    return prompts_data


def analyze_partial_solutions(prompts_data):
    """Analyze the partial solutions to understand what's being captured."""
    
    print("\n" + "=" * 60)
    print("PARTIAL SOLUTION ANALYSIS")
    print("=" * 60)
    
    creative_data = [d for d in prompts_data if d['label'] == 1]
    non_creative_data = [d for d in prompts_data if d['label'] == 0]
    
    # Analyze what techniques are visible in partial solutions
    def count_visible_techniques(partial_solution, machine_techniques):
        visible = []
        technique_keywords = {
            'for loop': 'for ',
            'while loop': 'while ',
            'if statement': 'if ',
            'list comprehension': ['[' and 'for' and 'in'],
            'lambda': 'lambda',
            'yield': 'yield',
            'generator': 'yield',
            'map': 'map(',
            'filter': 'filter(',
            'enumerate': 'enumerate(',
            'zip': 'zip(',
            'try except': 'try:',
            'with statement': 'with ',
            'recursion': 'def ',  # Approximate
        }
        
        for tech in machine_techniques:
            if tech in technique_keywords:
                keyword = technique_keywords[tech]
                if isinstance(keyword, str):
                    if keyword in partial_solution:
                        visible.append(tech)
                elif isinstance(keyword, list):
                    if all(k in partial_solution for k in keyword):
                        visible.append(tech)
        
        return visible
    
    # Analyze creative solutions
    creative_visible_ratios = []
    for d in creative_data:
        visible = count_visible_techniques(d['partial_solution'], d['machine_techniques'])
        ratio = len(visible) / max(len(d['machine_techniques']), 1)
        creative_visible_ratios.append(ratio)
    
    # Analyze non-creative solutions
    non_creative_visible_ratios = []
    for d in non_creative_data:
        visible = count_visible_techniques(d['partial_solution'], d['machine_techniques'])
        ratio = len(visible) / max(len(d['machine_techniques']), 1)
        non_creative_visible_ratios.append(ratio)
    
    print(f"Creative solutions - techniques visible in partial: {np.mean(creative_visible_ratios):.1%}")
    print(f"Non-creative solutions - techniques visible in partial: {np.mean(non_creative_visible_ratios):.1%}")
    
    # Character-based statistics
    all_char_percentages = [d['partial_percentage'] for d in prompts_data]
    creative_char_percentages = [d['partial_percentage'] for d in creative_data]
    non_creative_char_percentages = [d['partial_percentage'] for d in non_creative_data]
    
    print(f"\nCharacter-based Coverage:")
    print(f"  Overall average: {np.mean(all_char_percentages):.1%}")
    print(f"  Creative solutions: {np.mean(creative_char_percentages):.1%}")
    print(f"  Non-creative solutions: {np.mean(non_creative_char_percentages):.1%}")
    print(f"  Standard deviation: {np.std(all_char_percentages):.1%}")
    print(f"  Min: {np.min(all_char_percentages):.1%}, Max: {np.max(all_char_percentages):.1%}")
    
    # Line-based statistics for comparison
    all_line_percentages = [d['partial_line_percentage'] for d in prompts_data]
    print(f"\nLine-based Coverage (for comparison):")
    print(f"  Overall average: {np.mean(all_line_percentages):.1%}")
    
    # Character count statistics
    all_partial_chars = [d['partial_char_count'] for d in prompts_data]
    all_full_chars = [d['full_char_count'] for d in prompts_data]
    
    print(f"\nCharacter Counts:")
    print(f"  Average full solution: {np.mean(all_full_chars):.0f} chars")
    print(f"  Average partial solution: {np.mean(all_partial_chars):.0f} chars")
    
    # Distribution analysis
    print(f"\nPercentage Distribution:")
    percentile_points = [10, 25, 50, 75, 90]
    percentiles = np.percentile(all_char_percentages, percentile_points)
    for p, v in zip(percentile_points, percentiles):
        print(f"  {p}th percentile: {v:.1%}")
    
    # How many are within 5% of target?
    target = percentage
    within_5_percent = sum(1 for p in all_char_percentages if abs(p - target) <= 0.05)
    print(f"\nSolutions within 5% of target ({percentage_name}): {within_5_percent}/{len(prompts_data)} ({within_5_percent/len(prompts_data):.1%})")


def save_datasets(prompts_data, test_size=0.2, val_size=0.2):
    """Split and save the dataset for ITI training."""
    
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
    
    # Calculate average partial percentages (character-based)
    avg_partial_pct = np.mean([d['partial_percentage'] for d in prompts_data])
    print(f"\nAverage partial solution percentage (character-based): {avg_partial_pct:.1%}")
    
    # Save datasets
    Path('creativity_data_partial').mkdir(exist_ok=True)
    
    with open('creativity_data_partial/train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('creativity_data_partial/val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open('creativity_data_partial/test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # Also save as JSON for readability
    with open('creativity_data_partial/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('creativity_data_partial/val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
        
    with open('creativity_data_partial/test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("\n✓ Datasets saved to creativity_data_partial/")
    
    return train_data, val_data, test_data


def main():
    """Main function to create partial solution dataset."""
    
    # Analyze the data
    df = analyze_creativity_data()
    
    # Create balanced dataset
    balanced_df = prepare_balanced_dataset(df, creativity_threshold=0.1, max_samples_per_class=40)
    
    # Convert to prompts with 30% solutions (character-based)
    prompts_data = create_prompts_with_partial_solutions(balanced_df, percentage=percentage)
    
    # Analyze what's captured in partial solutions
    analyze_partial_solutions(prompts_data)
    
    # Save splits
    train_data, val_data, test_data = save_datasets(prompts_data)
    
    # Show examples
    print("\n" + "=" * 60)
    print(f"EXAMPLE: CREATIVE SOLUTION ({percentage_name} character-based)")
    print("=" * 60)
    creative_example = [d for d in train_data if d['label'] == 1][0]
    print(f"Problem ID: {creative_example['problem_id']}")
    print(f"New techniques ratio: {creative_example['new_techniques_ratio']:.2f}")
    print(f"Actual percentage shown (chars): {creative_example['partial_percentage']:.1%}")
    print(f"Actual percentage shown (lines): {creative_example['partial_line_percentage']:.1%}")
    print(f"Character count: {creative_example['partial_char_count']}/{creative_example['full_char_count']}")
    print(f"\n--- PARTIAL SOLUTION SHOWN ---")
    print(creative_example['partial_solution'])
    print(f"\n--- FULL SOLUTION ---")
    print(creative_example['full_solution'])
    print(f"\nTechniques: {creative_example['machine_techniques']}")
    
    print("\n" + "=" * 60)
    print(f"EXAMPLE: NON-CREATIVE SOLUTION ({percentage_name} character-based)")
    print("=" * 60)
    non_creative_example = [d for d in train_data if d['label'] == 0][0]
    print(f"Problem ID: {non_creative_example['problem_id']}")
    print(f"Actual percentage shown (chars): {non_creative_example['partial_percentage']:.1%}")
    print(f"Actual percentage shown (lines): {non_creative_example['partial_line_percentage']:.1%}")
    print(f"Character count: {non_creative_example['partial_char_count']}/{non_creative_example['full_char_count']}")
    print(f"\n--- PARTIAL SOLUTION SHOWN ---")
    print(non_creative_example['partial_solution'])
    print(f"\n--- FULL SOLUTION ---")
    print(non_creative_example['full_solution'])
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✓ Created {len(prompts_data)} prompts with character-based partial solutions")
    print(f"✓ Target coverage: {percentage_name} of code characters (excluding whitespace)")
    print(f"✓ Actual average coverage: {np.mean([d['partial_percentage'] for d in prompts_data]):.1%}")


if __name__ == "__main__":
    main()