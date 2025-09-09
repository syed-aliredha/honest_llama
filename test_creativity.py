# test_creativity_simple.py
import torch
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Set
import re
import warnings
warnings.filterwarnings('ignore')

from apply_creativity_iti import CreativityITI


class SimpleCreativityTester:
    """Simple tester for creativity ITI - just shows outputs"""
    
    def __init__(self, alpha=0.4, temperature=0.7):
        self.alpha = alpha
        self.temperature = temperature
        self.original_alpha = alpha
        
        # Initialize ITI model
        print(f"Loading model with α={alpha}...")
        self.iti_model = CreativityITI(alpha=alpha)
        self.iti_model.original_alpha = alpha
        
        # Load dataset
        self.df = pd.read_csv('Llama-8B-Instruct_dola3_creativity.csv')
        
    def extract_techniques(self, code: str) -> Set[str]:
        """Simple technique extraction"""
        techniques = set()
        
        # Define patterns
        patterns = {
            'list comprehension': r'\[.*\bfor\s+.*\bin\s+.*\]',
            'generator': r'\byield\s+',
            'lambda': r'\blambda\s+',
            'map': r'\bmap\s*\(',
            'filter': r'\bfilter\s*\(',
            'reduce': r'\breduce\s*\(',
            'enumerate': r'\benumerate\s*\(',
            'zip': r'\bzip\s*\(',
            'recursion': r'def\s+(\w+).*\n.*\1\s*\(',  # Simplified
            'try/except': r'\btry\s*:.*\bexcept',
            'with statement': r'\bwith\s+',
            'f-string': r'f["\'].*\{.*\}.*["\']',
            'walrus operator': r':=',
            'ternary': r'.*\bif\s+.*\belse\s+',
            'set operations': r'\bset\s*\(|\{.*\}',
            'any/all': r'\b(any|all)\s*\(',
            'sorted': r'\bsorted\s*\(',
            'min/max': r'\b(min|max)\s*\(',
        }
        
        for name, pattern in patterns.items():
            if re.search(pattern, code, re.DOTALL):
                techniques.add(name)
        
        # Basic patterns
        if 'for ' in code:
            techniques.add('for loop')
        if 'while ' in code:
            techniques.add('while loop')
        if 'if ' in code:
            techniques.add('if statement')
            
        return techniques
    
    def test_problems(self, num_problems=5):
        """Test on a few problems and show results"""
        
        # Get correct solutions only
        correct_df = self.df[self.df['correctness'] == True]
        
        # Sample problems
        sample_df = correct_df.sample(n=min(num_problems, len(correct_df)), random_state=42)
        
        print("\n" + "="*80)
        print(f"TESTING {num_problems} PROBLEMS WITH CREATIVITY ITI (α={self.alpha})")
        print("="*80)
        
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            problem_id = row['problem_id']
            constraints = eval(row['constraints']) if isinstance(row['constraints'], str) else row['constraints']
            human_techniques = eval(row['human_techniques']) if isinstance(row['human_techniques'], str) else row['human_techniques']
            original_techniques = eval(row['machine_techniques']) if isinstance(row['machine_techniques'], str) else row['machine_techniques']
            
            print(f"\n{'#'*80}")
            print(f"PROBLEM {idx}: {problem_id}")
            print(f"{'#'*80}")
            
            # Create problem prompt
            problem_text = f"""Write a Python function that must use these specific programming constructs:
{', '.join(constraints)}

The solution MUST include all of these elements."""
            print(f"\nProblem: {problem_text}")
            
            # Generate with ITI
            print(f"\n--- CREATIVE SOLUTION (ITI α={self.alpha}) ---")
            creative_solution = self.iti_model.generate_creative_solution(
                problem_text, 
                temperature=self.temperature
            )
            print(creative_solution)
            
            # Generate baseline for comparison
            print(f"\n--- BASELINE SOLUTION (No ITI) ---")
            self.iti_model.alpha = 0
            baseline_solution = self.iti_model.generate_creative_solution(
                problem_text,
                temperature=self.temperature
            )
            self.iti_model.alpha = self.original_alpha
            print(baseline_solution)
            
            # Analyze techniques
            creative_techniques = self.extract_techniques(creative_solution)
            baseline_techniques = self.extract_techniques(baseline_solution)
            new_creative_techniques = creative_techniques - baseline_techniques
            
            # Compare to human techniques
            human_set = set(human_techniques)
            new_vs_human = creative_techniques - human_set
            
            print(f"\n--- TECHNIQUE ANALYSIS ---")
            print(f"Human techniques: {human_techniques}")
            print(f"Original machine techniques: {original_techniques}")
            print(f"\nBaseline generated techniques: {list(baseline_techniques)}")
            print(f"Creative generated techniques: {list(creative_techniques)}")
            print(f"\nNEW techniques in creative vs baseline: {list(new_creative_techniques) if new_creative_techniques else 'None'}")
            print(f"NEW techniques vs human solutions: {list(new_vs_human) if new_vs_human else 'None'}")
            
            # Show creativity metrics from original data
            print(f"\nOriginal creativity ratio: {row['new_techniques_ratio']:.2f}")
            
            if idx < num_problems:
                input("\nPress Enter to see next problem...")


def main():
    parser = argparse.ArgumentParser(description="Simple test of Creativity ITI")
    parser.add_argument('--alpha', type=float, default=0.4, help='ITI intervention strength')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--num_problems', type=int, default=5, help='Number of problems to test')
    
    args = parser.parse_args()
    
    # Create tester and run
    tester = SimpleCreativityTester(alpha=args.alpha, temperature=args.temperature)
    tester.test_problems(num_problems=args.num_problems)


if __name__ == "__main__":
    main()