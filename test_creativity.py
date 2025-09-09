# test_creativity.py
"""
Test creativity intervention by evaluating the model's ability to generate
solutions that avoid common techniques while still being correct.
Following NeoCoder's approach to divergent creativity evaluation.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import List, Set, Dict, Tuple
import re
import ast
import json
from collections import Counter
import pandas as pd


class CreativityTester:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                 intervention_dir="creativity_iti_components",
                 device="cuda"):
        """
        Initialize the creativity tester with ITI intervention.
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load intervention components
        self.load_intervention_components(intervention_dir)
        
    def load_intervention_components(self, intervention_dir):
        """Load ITI intervention directions and settings."""
        intervention_path = Path(intervention_dir)
        
        # Check if intervention components exist
        if not intervention_path.exists():
            print(f"Warning: No intervention components found at {intervention_path}")
            print("Running without ITI intervention")
            self.top_heads = []
            self.directions = {}
            return
        
        try:
            with open(intervention_path / 'probe_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
            self.top_heads = results['top_heads']
            self.directions = results['directions']
            self.num_layers = results['num_layers']
            self.num_heads = results['num_heads']
            
            print(f"Loaded interventions for {len(self.top_heads)} heads")
        except Exception as e:
            print(f"Warning: Could not load interventions: {e}")
            self.top_heads = []
            self.directions = {}
    
    def load_original_dataset(self):
        """Load the original dataset with full problems."""
        csv_path = 'Llama-8B-Instruct_dola3_creativity.csv'
        if Path(csv_path).exists():
            return pd.read_csv(csv_path)
        else:
            print(f"Warning: Could not find {csv_path}")
            return None
    
    def extract_techniques(self, solution: str) -> Set[str]:
        """
        Extract programming techniques/patterns from a solution.
        """
        techniques = set()
        
        # Clean the solution
        solution_lower = solution.lower()
        
        # Common algorithmic patterns
        patterns = {
            'nested_loop': r'for.*:\s*\n\s*for.*:',
            'while_loop': r'while\s+.*:',
            'list_comprehension': r'\[.*for.*in.*\]',
            'generator': r'\(.*for.*in.*\)',
            'lambda': r'lambda\s+.*:',
            'map_function': r'map\s*\(',
            'filter_function': r'filter\s*\(',
            'enumerate': r'enumerate\s*\(',
            'zip_function': r'zip\s*\(',
            'dictionary_comprehension': r'\{.*:.*for.*in.*\}',
            'set_comprehension': r'\{.*for.*in.*\}',
            'try_except': r'try:.*except',
            'with_statement': r'with\s+.*:',
            'f_string': r'f["\'].*\{.*\}.*["\']',
            'counter': r'Counter\s*\(',
            'defaultdict': r'defaultdict\s*\(',
            'recursion': r'def\s+\w+\(.*\).*\n.*\w+\(',  # Simple recursion detection
            'slice_notation': r'\[.*:.*\]',
            'ternary_operator': r'.*if.*else.*',
        }
        
        for technique, pattern in patterns.items():
            if re.search(pattern, solution_lower, re.DOTALL):
                techniques.add(technique)
        
        # Simple keyword checks
        if 'sort(' in solution_lower or 'sorted(' in solution_lower:
            techniques.add('sorting')
        if '.reverse()' in solution_lower or 'reversed(' in solution_lower:
            techniques.add('reversing')
        if 'min(' in solution_lower or 'max(' in solution_lower:
            techniques.add('min_max')
        if 'sum(' in solution_lower:
            techniques.add('summation')
        if 'range(' in solution_lower:
            techniques.add('range_iteration')
        if '%' in solution:
            techniques.add('modulo')
        if '//' in solution:
            techniques.add('integer_division')
        if 'math.' in solution_lower or 'import math' in solution_lower:
            techniques.add('math_library')
        if 'random.' in solution_lower or 'import random' in solution_lower:
            techniques.add('random_library')
        if '.split(' in solution_lower:
            techniques.add('string_split')
        if '.join(' in solution_lower:
            techniques.add('string_join')
        if 'set(' in solution_lower:
            techniques.add('set_usage')
        if 'dict(' in solution_lower or '{' in solution and ':' in solution:
            techniques.add('dict_usage')
        
        return techniques
    
    def generate_baseline_solution(self, problem_text: str, constraints: List[str],
                                  temperature: float = 0.3) -> Tuple[str, Set[str]]:
        """Generate a baseline solution without technique constraints."""
        
        constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else ""
        
        prompt = f"""Problem: {problem_text}

{'Constraints:' if constraint_text else ''}
{constraint_text}

Write a Python function to solve this problem:

```python"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract code
        if "```" in generated:
            solution = generated.split("```")[0].strip()
        else:
            solution = generated.strip()
        
        techniques = self.extract_techniques(solution)
        return solution, techniques
    
    def generate_creative_solution(self, problem_text: str, constraints: List[str],
                                  avoid_techniques: List[str], 
                                  alpha: float = 15.0, temperature: float = 0.7) -> Tuple[str, Set[str], bool, Set[str]]:
        """
        Generate a solution that avoids specified techniques.
        """
        
        constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else ""
        avoid_text = "\n".join(f"- Do NOT use {tech.replace('_', ' ')}" for tech in avoid_techniques)
        
        prompt = f"""Problem: {problem_text}

{'Original Constraints:' if constraint_text else ''}
{constraint_text}

IMPORTANT ADDITIONAL CONSTRAINTS - You must solve this problem WITHOUT using:
{avoid_text}

Instead, find a creative, unconventional approach that still correctly solves the problem.
Think outside the box and use alternative methods.

Write a Python function:

```python"""
        
        # Generate with intervention (simplified - full implementation would apply actual ITI)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract code
        if "```" in generated:
            solution = generated.split("```")[0].strip()
        else:
            solution = generated.strip()
        
        # Check which techniques were used
        used_techniques = self.extract_techniques(solution)
        avoided_techniques_set = set(avoid_techniques)
        
        # Check if successfully avoided constraints
        violated_constraints = used_techniques.intersection(avoided_techniques_set)
        avoided_successfully = len(violated_constraints) == 0
        
        return solution, used_techniques, avoided_successfully, violated_constraints
    
    def evaluate_divergent_creativity(self, test_problems: List[Dict], 
                                     alpha: float = 15.0) -> Dict:
        """
        Evaluate the model's ability to generate diverse solutions.
        """
        results = {
            'problems': [],
            'overall_metrics': {},
            'technique_distribution': Counter(),
            'constraint_violations': Counter()
        }
        
        # Load original dataset if available
        df = self.load_original_dataset()
        
        for idx, problem_data in enumerate(test_problems):
            # Extract problem info
            problem_id = problem_data.get('problem_id', f'problem_{idx}')
            
            # Get full problem description from original dataset if available
            if df is not None and problem_id in df['problem_id'].values:
                row = df[df['problem_id'] == problem_id].iloc[0]
                problem_text = row.get('problem_description', '')
                constraints_str = row.get('constraints', '[]')
                try:
                    constraints = eval(constraints_str) if isinstance(constraints_str, str) else constraints_str
                except:
                    constraints = []
            else:
                # Fallback to extracting from prompt
                prompt = problem_data.get('prompt', '')
                # Extract problem description
                if "Problem " in prompt:
                    match = re.search(r'Problem\s+\w+:(.*?)(?:Write|Constraints:|Here is)', prompt, re.DOTALL)
                    problem_text = match.group(1).strip() if match else prompt.split('\n')[0]
                else:
                    problem_text = prompt.split('\n')[0]
                
                constraints = problem_data.get('constraints', [])
            
            is_creative = problem_data.get('is_creative', problem_data.get('label', 0))
            
            print(f"\n{'='*60}")
            print(f"PROBLEM {problem_id}")
            print(f"{'='*60}")
            print(f"\n📝 QUESTION:")
            print(f"{problem_text}")
            
            if constraints:
                print(f"\n📋 CONSTRAINTS:")
                for c in constraints:
                    print(f"  • {c}")
            
            print(f"\n🎯 DATASET LABEL: {'Creative' if is_creative else 'Non-creative'}")
            
            # Generate baseline solution first
            print(f"\n🔧 Generating baseline solution...")
            baseline_solution, baseline_techniques = self.generate_baseline_solution(
                problem_text, constraints, temperature=0.3
            )
            
            print(f"\n📊 BASELINE SOLUTION:")
            print("```python")
            print(baseline_solution)
            print("```")
            print(f"Techniques used: {list(baseline_techniques)[:7]}")
            
            # Select techniques to avoid (top 3-5 most common)
            techniques_to_avoid = list(baseline_techniques)[:min(4, len(baseline_techniques))]
            
            print(f"\n🚫 TECHNIQUES TO AVOID:")
            if techniques_to_avoid:
                for tech in techniques_to_avoid:
                    print(f"  • {tech.replace('_', ' ')}")
            else:
                print("  • None identified")
            
            # Generate creative solution avoiding those techniques
            print(f"\n✨ Generating creative solution (avoiding common techniques)...")
            creative_solution, used_techniques, avoided_successfully, violations = \
                self.generate_creative_solution(
                    problem_text, 
                    constraints,
                    techniques_to_avoid, 
                    alpha
                )
            
            print(f"\n🎨 CREATIVE SOLUTION:")
            print("```python")
            print(creative_solution)
            print("```")
            
            print(f"\n📈 ANALYSIS:")
            print(f"  • Techniques used: {list(used_techniques)[:7]}")
            print(f"  • Successfully avoided constraints: {'✅ Yes' if avoided_successfully else '❌ No'}")
            if violations:
                print(f"  • Violated constraints: {list(violations)}")
            
            # Calculate divergence score (how different from baseline)
            new_techniques = used_techniques - baseline_techniques
            shared_techniques = used_techniques & baseline_techniques
            divergence_score = len(new_techniques) / (len(used_techniques) + 1e-6)
            
            print(f"  • New techniques introduced: {list(new_techniques)[:5]}")
            print(f"  • Divergence score: {divergence_score:.2%}")
            
            # Store results
            problem_result = {
                'problem_id': problem_id,
                'problem_text': problem_text,
                'constraints': constraints,
                'original_creative': bool(is_creative),
                'baseline_solution': baseline_solution,
                'baseline_techniques': list(baseline_techniques),
                'creative_solution': creative_solution,
                'avoided_techniques': techniques_to_avoid,
                'used_techniques': list(used_techniques),
                'avoided_successfully': avoided_successfully,
                'violations': list(violations),
                'new_techniques': list(new_techniques),
                'divergence_score': divergence_score
            }
            results['problems'].append(problem_result)
            
            # Update counters
            results['technique_distribution'].update(used_techniques)
            results['constraint_violations'].update(violations)
        
        # Calculate overall metrics
        total = len(results['problems'])
        successful = sum(1 for p in results['problems'] if p['avoided_successfully'])
        
        # Separate metrics for originally creative vs non-creative
        creative_problems = [p for p in results['problems'] if p['original_creative']]
        non_creative_problems = [p for p in results['problems'] if not p['original_creative']]
        
        avg_divergence = np.mean([p['divergence_score'] for p in results['problems']])
        
        results['overall_metrics'] = {
            'total_problems': total,
            'successful_avoidance': successful,
            'success_rate': successful / total if total > 0 else 0,
            'unique_techniques': len(results['technique_distribution']),
            'avg_techniques_per_solution': sum(len(p['used_techniques']) 
                                              for p in results['problems']) / total if total > 0 else 0,
            'avg_divergence_score': avg_divergence,
            'creative_problems_success_rate': sum(1 for p in creative_problems if p['avoided_successfully']) / len(creative_problems) if creative_problems else 0,
            'non_creative_problems_success_rate': sum(1 for p in non_creative_problems if p['avoided_successfully']) / len(non_creative_problems) if non_creative_problems else 0
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test creativity using divergent thinking approach")
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help='Model to test')
    parser.add_argument('--alpha', type=float, default=15.0, 
                       help='ITI intervention strength')
    parser.add_argument('--num_problems', type=int, default=5,
                       help='Number of problems to test')
    parser.add_argument('--test_file', type=str, default='creativity_data_partial/test.pkl',
                       help='Test data file (for problem IDs)')
    parser.add_argument('--output', type=str, default='creativity_evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load test problems
    test_file = Path(args.test_file)
    if test_file.suffix == '.pkl':
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
    else:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
    
    # Sample problems if needed
    if args.num_problems < len(test_data):
        import random
        random.seed(42)
        # Try to get balanced creative/non-creative
        creative = [d for d in test_data if d.get('label', d.get('is_creative', 0))]
        non_creative = [d for d in test_data if not d.get('label', d.get('is_creative', 0))]
        
        n_each = args.num_problems // 2
        test_data = random.sample(creative, min(n_each, len(creative))) + \
                   random.sample(non_creative, min(args.num_problems - n_each, len(non_creative)))
        random.shuffle(test_data)
    
    # Initialize tester
    tester = CreativityTester(model_name=args.model)
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"EVALUATING DIVERGENT CREATIVITY")
    print(f"Model: {args.model}")
    print(f"Alpha: {args.alpha}")
    print(f"Testing on {len(test_data)} problems")
    print(f"{'='*60}")
    
    results = tester.evaluate_divergent_creativity(test_data, alpha=args.alpha)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    metrics = results['overall_metrics']
    print(f"Total problems: {metrics['total_problems']}")
    print(f"Successfully avoided common techniques: {metrics['successful_avoidance']}")
    print(f"Overall success rate: {metrics['success_rate']:.2%}")
    print(f"Average divergence score: {metrics['avg_divergence_score']:.2%}")
    print(f"Success rate on creative problems: {metrics['creative_problems_success_rate']:.2%}")
    print(f"Success rate on non-creative problems: {metrics['non_creative_problems_success_rate']:.2%}")
    print(f"Unique techniques used: {metrics['unique_techniques']}")
    print(f"Average techniques per solution: {metrics['avg_techniques_per_solution']:.2f}")
    
    print(f"\nMost common techniques used:")
    for tech, count in results['technique_distribution'].most_common(10):
        print(f"  {tech}: {count}")
    
    if results['constraint_violations']:
        print(f"\nMost violated constraints:")
        for tech, count in results['constraint_violations'].most_common(5):
            print(f"  {tech}: {count}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()