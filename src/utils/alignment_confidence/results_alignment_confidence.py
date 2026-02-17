import json
import os

from together import Together
from dotenv import load_dotenv
load_dotenv()

import json
import re

VALID_ALIGNMENTS = {
    "Perfectly Aligned",
    "Aligned",
    "Slightly Misaligned",
    "Misaligned"
}

def parse_output(raw_output: str) -> dict:
    """
    Parse LLM alignment evaluation output.

    Expected JSON format:
    {
      "implied_confidence_class": "...",
      "alignment_judgment": "Perfectly Aligned | Aligned | Slightly Misaligned | Misaligned",
      "justification": "..."
    }

    Returns:
        dict with parsed fields

    Raises:
        ValueError if parsing or validation fails.
    """

    if not raw_output or not raw_output.strip():
        raise ValueError("Empty output.")

    # Extract JSON block (handles ```json ... ``` or raw JSON)
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in output.")

    json_str = json_match.group(0)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Validate required keys
    required_keys = {
        "implied_confidence_class",
        "alignment_judgment",
        "justification"
    }

    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Validate alignment judgment
    if parsed["alignment_judgment"] not in VALID_ALIGNMENTS:
        raise ValueError(
            f"Invalid alignment_judgment: {parsed['alignment_judgment']}"
        )

    # Optional: strip whitespace
    parsed["implied_confidence_class"] = parsed["implied_confidence_class"].strip()
    parsed["justification"] = parsed["justification"].strip()

    return parsed



class_labels = [
    "zero certainty",
    "minimal certainty",
    "very low certainty",
    "low certainty",
    "low-moderate certainty",
    "moderate certainty",
    "moderate-high certainty",
    "high certainty",
    "very high certainty",
    "near-absolute certainty"
]

conf2score = {
    "zero certainty": 0.05,
    "minimal certainty": 0.15,
    "very low certainty": 0.25,
    "low certainty": 0.35,
    "low-moderate certainty": 0.45,
    "moderate certainty": 0.55,
    "moderate-high certainty": 0.65,
    "high certainty": 0.75,
    "very high certainty": 0.85,
    "near-absolute certainty": 0.95
}


def save_results_together(job_name, output_dir):
    batch_stat = client.batches.get_batch(job_name)

    print(batch_stat.status)
    # # Get the batch status to find output_file_id
    batch = client.batches.get_batch(job_name)
    output_file = output_dir + "/raw_completions.jsonl"

    if batch.status == "COMPLETED":
       
        # Download the output file
        client.files.retrieve_content(
            id=batch_stat.output_file_id,
            output=output_file,
        )

        with open(output_file) as f:
            completions = [json.loads(line) for line in f.readlines()]
        
        for item in completions:
            parts = item['custom_id'].split("_", 1)
            id_item = parts[1]

            completion = item['response']['body']['choices'][0]['message']['content']
            completion_length = item['response']['body']['usage']['completion_tokens']
            final_answer = ""
            reasoning = ""
            
            
            
            reasoning = item['response']['body']['choices'][0]['message']['reasoning']
            if completion.strip():
                output = parse_output(completion)
            else:
                output = parse_output(completion)     
        
            if output:
                
                implied_confidence_class = output["implied_confidence_class"]
                alignment_judgment = output["alignment_judgment"]   
                justification = output["justification"]
            
            with open(f"{output_dir}/generations.jsonl", "a") as f:
                json.dump({"id_question": id_item, 
                            "implied_confidence_class": implied_confidence_class, 
                            "alignment_judgment": alignment_judgment, 
                            "justification": justification,
                            "completion_length": completion_length}, f)
                f.write("\n")
    else:
        print(f"Batch not completed. Current status: {batch.status}")

if __name__ == "__main__":
    
    client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
    print("Processing results...")
    job_name = "f350dd81-d31a-4d65-b293-129314a27ce7"
    output_dir = "out/alignment_confidence/medgemma/medxpertqa"
    save_results_together(job_name, output_dir)
    print("Done!")


   
   