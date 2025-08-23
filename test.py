# Test script to verify configuration
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your modified files
import sys
sys.path.append('get_activations/')
from get_activations import HF_NAMES

# Check if llama3.1_8B_instruct is in the dictionary
assert 'llama3.1_8B_instruct' in HF_NAMES
print(f"✓ Model path configured: {HF_NAMES['llama3.1_8B_instruct']}")

# Try loading the tokenizer to verify the path is correct
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_NAMES['llama3.1_8B_instruct'])
    print("✓ Successfully loaded tokenizer")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("You may need to authenticate with HuggingFace or request access to the model")