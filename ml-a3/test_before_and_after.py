import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import sys
import os
import random

# CONFIGURATION
BASE_MODEL_PATH = "./Qwen2.5-1.5B-Instruct"
TUNED_MODEL_PATH = "./saved_model"
TEST_DATA_PATH = "gsm8k/main/test-00000-of-00001.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 5  # How many questions to test

def load_test_data(filepath, n=3):
    """Loads specific parquet file and samples n questions"""
    print(f"Loading test data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        # Sample n random rows
        sample_df = df.sample(n=n)
        return sample_df.to_dict('records') # Returns list of dicts: [{'question':..., 'answer':...}, ...]
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        print("Make sure pandas and pyarrow are installed: pip install pandas pyarrow")
        sys.exit(1)

def load_model(path, device):
    """Helper to load model and tokenizer efficiently"""
    print(f"Loading model from: {path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        sys.exit(1)

def generate_answer(model, tokenizer, question, device):
    """Generates a response for a single question"""
    # GSM8K formatting used in training
    prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,       # Deterministic for fair comparison
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt to see just the answer
    answer_only = full_text.replace(prompt, "").strip()
    return answer_only

def main():
    print(f"Running comparison on device: {DEVICE}")
    print("=" * 60)

    # 1. Load Data
    test_samples = load_test_data(TEST_DATA_PATH, n=NUM_SAMPLES)

    # 2. Test Base Model
    print("\n>>> PHASE 1: Testing Base Model (Before Training)")
    base_model, base_tokenizer = load_model(BASE_MODEL_PATH, DEVICE)
    
    base_outputs = []
    for item in test_samples:
        ans = generate_answer(base_model, base_tokenizer, item['question'], DEVICE)
        base_outputs.append(ans)
    
    # Free up memory
    del base_model
    torch.cuda.empty_cache()
    print("Base model unloaded.\n")

    # 3. Test Tuned Model
    print(">>> PHASE 2: Testing Tuned Model (After GRPO Training)")
    tuned_model, tuned_tokenizer = load_model(TUNED_MODEL_PATH, DEVICE)
    
    tuned_outputs = []
    for item in test_samples:
        ans = generate_answer(tuned_model, tuned_tokenizer, item['question'], DEVICE)
        tuned_outputs.append(ans)

    # 4. Print Side-by-Side Report
    print("\n" + "="*80)
    print(f"{'BEFORE / AFTER COMPARISON REPORT':^80}")
    print("="*80)
    
    for i, item in enumerate(test_samples):
        print(f"\n[TEST CASE {i+1}]")
        print(f"QUESTION: {item['question']}")
        print("-" * 80)
        print(f"GROUND TRUTH ANSWER:\n{item['answer']}")
        print("-" * 80)
        print(f"[BASE MODEL OUTPUT]:\n{base_outputs[i]}")
        print("-" * 40)
        print(f"[GRPO MODEL OUTPUT]:\n{tuned_outputs[i]}")
        print("=" * 80)

if __name__ == "__main__":
    main()