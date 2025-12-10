import json
import argparse
from google import genai
from google.genai import types
from openai import OpenAI
from together import Together
import time
from dotenv import load_dotenv
import os
load_dotenv()


import re

# - Zero Certainty (0.0-0.1)
    
#     - Minimal Certainty (0.1-0.2)

#     - Very Low Certainty (0.2-0.3)

#     - Low Certainty (0.3-0.4)

#     - Low-Moderate Certainty (0.4-0.5)

#     - Moderate Certainty (0.5-0.6)

#     - Moderate-High Certainty (0.6-0.7)

#     - High Certainty (0.7-0.8)

#     - Very High Certainty (0.8-0.9)
 
#     - Near-Absolute Certainty (0.9-1.0)

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

def parse_output(text: str):
    """
    Parses output of the form:
        The final answer is \\boxed{<OPTION LETTER>}.
        Confidence: <CERTAINTY CLASS NAME>

    Returns:
        dict with keys "answer" and "confidence".
    """
    # Match \boxed{A}
    answer_match = re.search(r"\\boxed\{([A-Z])\}", text)
    # Match confidence after "Confidence:"
    conf_match = re.search(r"Confidence:\s*([A-Za-z\- ]+)", text)
    # check if confidence is in class_labels
    confidence = None
    if conf_match:
        confidence = conf_match.group(1).strip().lower()
        
        if "certainty" not in confidence:
            confidence += " certainty"

        if confidence not in class_labels:
            confidence = None

    return {
        "answer": answer_match.group(1) if answer_match else None,
        "confidence": conf_match.group(1).strip() if conf_match else None,
        "confidence_score": conf2score[confidence] if confidence in conf2score else None
    }


def save_results_gemini(job_name, output_dir, gold_answers=None):

    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30) # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")


    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step

    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")

            print("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            print(file_content.decode('utf-8'))

            with open(f"{output_dir}/raw_completions.jsonl", "w", encoding="utf-8") as f:
                f.write(file_content.decode("utf-8"))

            with open(f"{output_dir}/raw_completions.jsonl") as f:
                completions = [json.loads(line) for line in f.readlines()]
            
            
            for item in completions:
                key = item['key']
                if "MM" in key:
                    key = key.replace("MM-MM", "MM")
                key_splits = key.split("-", 1)
                subset = key_splits[0]
                
                id_item = key_splits[1]
                gold_answer = gold_answers[id_item] if gold_answers and id_item in gold_answers else None
                #parts = item['response']['candidates'][0]['content']['parts'] if 'candidates' in item['response'] and 'content' in item['response']['candidates'][0] else []
                parts = (
                    item.get('response', {})
                        .get('candidates', [{}])[0]
                        .get('content', {})
                        .get('parts', [])
                )

                if "no-think" in output_dir.lower():
                    #print(item['response']['usageMetadata'])
                    token_usage = item['response']['usageMetadata']['candidatesTokenCount'] if 'candidatesTokenCount' in item['response']['usageMetadata'] else None
                else:
                    token_usage = item['response']['usageMetadata']['thoughtsTokenCount'] if 'thoughtsTokenCount' in item['response']['usageMetadata'] else None

                thinking = ""
                answer = ""
                final_answer = ""
                for part in parts:
                    if 'thought' in part:
                        thinking = part['text']
                    else:
                        answer = part['text']
                        output = parse_output(answer)
                        final_answer = output['answer'] 
                        confidence = output['confidence']
                        confidence_score = output['confidence_score']
                
                with open(f"{output_dir}/generations_{subset}.jsonl", "a") as f:
                    json.dump({"id_question": id_item, "dataset": subset,  "gold_answer": gold_answer, "final_answer": final_answer, "confidence": confidence, "confidence_score": confidence_score, "correct": gold_answer == final_answer, "completion": answer, "thinking": thinking, "thinking_tokens": token_usage}, f)
                    f.write("\n")
            

        # If batch job was created with inline request
        # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
        elif batch_job.dest and batch_job.dest.inlined_responses:
            # Results are inline
            print("Results are inline:")
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                print(f"Response {i+1}:")
                if inline_response.response:
                    # Accessing response, structure may vary.
                    try:
                        print(inline_response.response.text)
                    except AttributeError:
                        print(inline_response.response) # Fallback
                elif inline_response.error:
                    print(f"Error: {inline_response.error}")
        else:
            print("No results found (neither file nor inline).")
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")


def save_results_openai(job_name, output_dir):

    #print(client.batches.retrieve(args.batch_id))
    response = client.batches.retrieve(job_name)
    is_safe = False
    if response.status =='completed':
        print("INFERENCE COMPLETED!")
        print(response)

        if response.error_file_id:
            print("ERROR FILE ID: ", response.error_file_id)
            file_response = client.files.content(response.error_file_id)
            for line in file_response.text.splitlines():
                print(line)
        elif response.output_file_id:
            is_safe = True
            print("OUTPUT FILE ID: ", response.output_file_id)
            file_response = client.files.content(response.output_file_id)
            out_file_id = response.output_file_id
        
            for line in file_response.text.splitlines():
                print(line)
            print("Saving results...")
            with open(f'{output_dir}/raw_completions.jsonl', 'w') as f:
                for line in file_response.text.splitlines():
                    json.dump(json.loads(line), f, ensure_ascii=False)
                    f.write('\n')
            print("Done!")

        if is_safe:
            print("Parsing results...")
            with open(f'{output_dir}/raw_completions.jsonl', 'r') as f:
                completions = [json.loads(line) for line in f.readlines()]
            
            
            for k, item in enumerate(completions):

                key = item['custom_id']
                if "MM" in key:
                    key = key.replace("MM-MM", "MM")
                key_splits = key.split("-", 1)
                subset = key_splits[0]
                
                id_item = key_splits[1]
                gold_answer = gold_answers[id_item] if gold_answers and id_item in gold_answers else None
                usage_info = item['response']['body']['usage']['output_tokens_details']['reasoning_tokens']
                #result = json.loads(line)
                output_request = item['response']['body']['output']
                final_answer = ""
                thinking = ""
                confidence = ""
                confidence_score = ""
                completion = ""
                for out in output_request:
                    
                    if out['type'] == "reasoning":
                        
                        if out['summary']:
                            for sum in out['summary']:
                                thinking += (sum['text'] + "\n\n")
                    elif out['type'] == "message":
                        completion = out['content'][0]['text'] if out['content'] else ""
                        #print(k, completion)
                        output = parse_output(completion)
                        print(k, output)
                        final_answer = output['answer'] 
                        confidence = output['confidence']
                        confidence_score = output['confidence_score']
                    
                        with open(f"{output_dir}/generations_{subset}.jsonl", "a") as f:
                            json.dump({"id_question": id_item, "dataset": subset,  "gold_answer": gold_answer, "final_answer": final_answer, "confidence": confidence, "confidence_score": confidence_score, "correct": gold_answer == final_answer, "completion": completion.strip(), "thinking": thinking.strip(), "thinking_tokens": usage_info}, f)
                            f.write("\n")
            print("Done!")

    else:
        print("BATCH STILL PROCESSING...")
        print(f"STATUS: {response.status}")
        if response.status == "failed":
            print(response)

def save_results_together(job_name, output_dir, gold_answers=None):
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
            parts = item['custom_id'].split("-", 1)
            
            subset = parts[0]
                
            id_item = parts[1]

            
            gold_answer = gold_answers[id_item] if gold_answers and id_item in gold_answers else None
            completion = item['response']['body']['choices'][0]['message']['content']
            completion_length = item['response']['body']['usage']['completion_tokens']
            final_answer = ""
            reasoning = ""
            if "gpt-oss" not in output_dir:
                #modes = ["incorrect", "none_of_the_provided", "options_only", "yes_no_maybe", "roman_numeral", "fixed_pos", "no_symbols"]

                final_answer_idx = completion.rfind("Final Answer:")
                if final_answer_idx > 0:
                    final_answer = completion[final_answer_idx:]
                    #reasoning = completion[:final_answer_idx]
                    output = parse_output(final_answer.replace("*",""))

            else:
                
                reasoning = item['response']['body']['choices'][0]['message']['reasoning']
                if completion.strip():
                    output = parse_output(completion)
                else:
                    output = parse_output(completion)     
        
            final_answer = output['answer'] 
            confidence = output['confidence']
            confidence_score = output['confidence_score']
            
            with open(f"{output_dir}/generations_{subset}.jsonl", "a") as f:
                json.dump({"id_question": id_item, "dataset": subset,  "gold_answer": gold_answer, "final_answer": final_answer, "confidence": confidence, "confidence_score": confidence_score, "correct": gold_answer == final_answer, "completion": completion, "thinking": reasoning, "thinking_tokens": completion_length}, f)
                f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Retrieve and save API batch job results.',
    )  

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions/together_api/openai/gpt-oss-120b/medqa_5opt/life-threatening/last/mask_question/2025-11-28_16-14-29",
        help="Output directory to save results."
    )

    parser.add_argument('--job-name', type=str, default="0a056eb9-ebf8-40ce-9adf-cfaffd9b2c09",
                      help='batch job id to retrieve results from')
    
    args = parser.parse_args()

    
    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = args.job_name #batches/tdw033ksy8zwa7e8spuj30075my7031b041i" #"batches/jdsea2vgjodl3ftrdvxgpp3f478wwteg01ld" #"batches/taxgnuxeblk3sq3hkgxc6pmskwbaoqtx74kt"  # (e.g. 'batches/your-batch-id')
    output_dir = args.output_dir #"out/legal/completions/gemini_api/gemini-2.5-flash/professional_law/2025-10-18_00-19-13"

    if "replace_gold" in output_dir:
        position_abstain = "replace_gold"
    elif "additional" in output_dir:
        position_abstain = "additional"
    elif "first" in output_dir:
        position_abstain = "first"
    else:
        position_abstain = "last"

    dataset_type = "LT" if "/life-threatening" in output_dir else "S"

    if "medqa" in output_dir:
        subset = "medqa_4opt" if "medqa_4opt" in output_dir else "medqa_5opt"
        data_path = f"data/bench/{subset}/{subset}_{dataset_type}.jsonl"
        
        with open(data_path, 'r') as f:
            benchmark = [json.loads(line) for line in f.readlines()]
        
        if position_abstain in ["replace_gold", "additional"]:
            gold_answers = {item['id']: item['answer_idx'] for item in benchmark}
        elif position_abstain == "first":
            gold_answers = {item['id']: "A" for item in benchmark}
        else:  # last
            gold_answers = {item['id']: chr(ord('A') + len(item['options']) - 1) for item in benchmark}

    elif "medmcqa" in output_dir:
        subset = "medmcqa"

        data_path = f"data/bench/{subset}/{subset}_{dataset_type}.jsonl"
        
        with open(data_path, 'r') as f:
            benchmark = [json.loads(line) for line in f.readlines()]
        
        benchmark = [{**item, 'options': {"A": item['opa'], "B": item['opb'], "C": item['opc'], "D": item['opd']}} for item in benchmark]
        num2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        benchmark = [{**item, "answer": num2letter[item['cop']]} for item in benchmark]

        if position_abstain in ["replace_gold", "additional"]:
            gold_answers = {item['id']: item['answer'] for item in benchmark}
        elif position_abstain == "first":
            gold_answers = {item['id']: "A" for item in benchmark}
        else:  # last
            gold_answers = {item['id']: chr(ord('A') + len(item['options']) - 1) for item in benchmark}
        
            
    elif "medxpertqa" in output_dir:
        subset = "medxpertqa-MM" if "MM" in output_dir else "medxpertqa"
        data_path = f"data/bench/{subset}/{subset}_{dataset_type}.jsonl"
        
        with open(data_path, 'r') as f:
            benchmark = [json.loads(line) for line in f.readlines()]

        if position_abstain in ["replace_gold", "additional"]:
            gold_answers = {item['id']: item['answer'] for item in benchmark}
        elif position_abstain == "first":
            gold_answers = {item['id']: "A" for item in benchmark}
        else:  # last
            gold_answers = {item['id']: chr(ord('A') + len(item['options']) - 1) for item in benchmark}
        
    else:
        raise ValueError("Subset not found in output directory path.")
    
    if "gemini" in output_dir:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)
        #batch_job = client.batches.get(name=job_name)
        print("Processing results...")
        save_results_gemini(job_name, output_dir, gold_answers=gold_answers)
        print("Done!")

    elif "together" in output_dir:
        client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
        print("Processing results...")
        save_results_together(job_name, output_dir, gold_answers=gold_answers)
        print("Done!")

    elif "openai" in output_dir:
        OPENAI_KEY = os.getenv("OPENAI_KEY")
        client = OpenAI(api_key=OPENAI_KEY)
        print("Processing results...")
        save_results_openai(job_name, output_dir)
        print("output_dir:", output_dir)
        print("Done!")

    else:
        print("Unknown model in output directory path.")

   