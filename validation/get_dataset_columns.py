#!/usr/bin/env python3

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("truthful_qa", "multiple_choice")['validation']

print("Available columns:", dataset.column_names)
print("Number of questions:", len(dataset))

# Show first example
print("\nFirst example:")
for key, value in dataset[0].items():
    if isinstance(value, list):
        print(f"  {key}: {value[:3]}..." if len(value) > 3 else f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")

# Check if there's a generation dataset that might have categories
print("\n" + "="*50)
print("Checking generation dataset for categories...")

try:
    gen_dataset = load_dataset("truthful_qa", "generation")['validation']
    print("Generation dataset columns:", gen_dataset.column_names)
    
    if 'category' in gen_dataset.column_names:
        print("✅ Found 'category' in generation dataset!")
        print("First 5 categories:", gen_dataset['category'][:5])
    else:
        print("❌ No 'category' in generation dataset either")
        
except Exception as e:
    print(f"Error loading generation dataset: {e}")