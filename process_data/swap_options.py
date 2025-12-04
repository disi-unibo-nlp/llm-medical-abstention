import json
import random
from pathlib import Path
from typing import List, Dict
import copy
import string

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def swap_mode_group(benchmark: List[Dict], seed: int = 42, answer_idx_key: str = "answer_idx", answer_key: str = "answer") -> List[Dict]:
    """
    Mode 1: Swap complete option groups between questions.
    Each question gets another question's complete set of 4 options.
    
    Args:
        benchmark: List of question dictionaries
        seed: Random seed for reproducibility
    
    Returns:
        List of questions with swapped option groups
    """
    random.seed(seed)
    
    # Create a deep copy
    swapped_data = copy.deepcopy(benchmark)
    
    # Collect all option sets
    all_options = []
    for item in benchmark:
        if 'options' in item and item['options']:
            all_options.append(item['options'])
    
    # Shuffle the option sets
    shuffled_options = all_options.copy()
    random.shuffle(shuffled_options)
    
    # Assign shuffled option sets to questions
    option_idx = 0
    for item in swapped_data:
        if 'options' in item and item['options']:
            # Assign new options (complete set from another question)
            item['options'] = shuffled_options[option_idx]
            
            # Remove answer_idx and answer since they're now invalid
            #item.pop(answer_idx_key, None)
            #item.pop(answer_key, None)

            # Instead of removing gold labels → assign a random letter
            num_options = len(item['options'])
            # Generate letters: A, B, C, D, ...
            letters = list(string.ascii_uppercase[:num_options])

            # Pick a random letter
            random_letter = random.choice(letters)
            # Save as new answer index and answer letter
            item[answer_idx_key] = random_letter
            item[answer_key] = item['options'][random_letter]

            
            option_idx += 1
    
    return swapped_data

def swap_mode_mix(benchmark: List[Dict], seed: int = 42, answer_idx_key: str = "answer_idx", answer_key: str = "answer") -> List[Dict]:
    """
    Mode 2: Mix individual options across all questions.
    Each question gets 4 random options from the pool of all options.
    
    Args:
        benchmark: List of question dictionaries
        seed: Random seed for reproducibility
    
    Returns:
        List of questions with mixed options
    """
    random.seed(seed)
    
    # Create a deep copy
    swapped_data = copy.deepcopy(benchmark)
    
    # Collect all individual options from all questions
    all_individual_options = []
    for item in benchmark:
        if 'options' in item and item['options']:
            for key, value in item['options'].items():
                all_individual_options.append(value)
    
    # For each question, randomly select 4 options
    for item in swapped_data:
        if 'options' in item and item['options']:
            # Sample 4 random options without replacement
            selected_options = random.sample(all_individual_options, 4)
            
            # Assign them to A, B, C, D
            item['options'] = {
                'A': selected_options[0],
                'B': selected_options[1],
                'C': selected_options[2],
                'D': selected_options[3]
            }
            
            # Remove answer_idx and answer since they're now invalid
            item.pop(answer_idx_key, None)
            item.pop(answer_key, None)
    
    return swapped_data

def main():
    # Configuration
    input_dir = "data/bench"  # Change this to your input directory
    output_dir = "data/swap"  # Change this to your output directory
    subset = "medxpertqa"  

    if "medqa" in subset:
        answer_key = "answer"
        answer_idx_key = "answer_idx"
    elif "medxpertqa" in subset:
        answer_key = ""
        answer_idx_key = "label"
    
    # Choose mode: "group" or "mix"
    mode = "group"  # Change to "mix" for mode 2
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data 
    data_path = f"{input_dir}/{subset}/{subset}_LT.jsonl"
    print(f"Loading data from: {data_path}")
    benchmark = load_jsonl(data_path)
    print(f"Loaded {len(benchmark)} questions")
    
    # Swap options based on mode
    print(f"Swapping options using mode: {mode}")
    if mode == "group":
        swapped_benchmark = swap_mode_group(benchmark, seed=42)
        suffix = "swapped_group"
    elif mode == "mix":
        swapped_benchmark = swap_mode_mix(benchmark, seed=42)
        suffix = "swapped_mix"
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'group' or 'mix'")
    
    # Save swapped data
    output_path = f"{output_dir}/{subset}_{suffix}.jsonl"
    save_jsonl(swapped_benchmark, output_path)
    print(f"Saved swapped data to: {output_path}")
    
    # Print examples
    print("\n=== Original Question ===")
    print(f"Question: {benchmark[0]['question'][:100]}...")
    print(f"Options: {benchmark[0]['options']}")
    print(f"Answer: {benchmark[0].get(answer_key, 'N/A')}")
    print(f"Answer Index: {benchmark[0].get(answer_idx_key, 'N/A')}")
    
    print(f"\n=== Swapped Question (Mode: {mode}) ===")
    print(f"Question: {swapped_benchmark[0]['question'][:100]}...")
    print(f"Options: {swapped_benchmark[0]['options']}")
    print(f"Answer: {swapped_benchmark[0].get(answer_key, 'REMOVED')}")
    print(f"Answer Index: {swapped_benchmark[0].get(answer_idx_key, 'REMOVED')}")
    
    print("\n" + "="*50)
    print(f"Mode '{mode}' explanation:")
    if mode == "group":
        print("- Each question gets a complete set of 4 options from another random question")
        print("- Options stay together as a group, just assigned to different questions")
    else:
        print("- Each question gets 4 random individual options from the entire pool")
        print("- Options are completely mixed across all questions")
    print("- Original answer and answer_idx are removed (no correct answers)")

if __name__ == "__main__":
    main()