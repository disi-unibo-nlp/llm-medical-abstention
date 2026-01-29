import json
import argparse
import os
import time
from google import genai
from google.genai import types
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
from src.utils.utils import get_gold_answers, parse_output, load_benchmark
load_dotenv()


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



def save_results_gemini(job_name, output_dir, gold_answers=None, subset=None):

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
                
                parts = (
                    item.get('response', {})
                        .get('candidates', [{}])[0]
                        .get('content', {})
                        .get('parts', [])
                )

                usage_metadata = (
                    item
                    .get("response", {})
                    .get("usageMetadata", {})
                )

                if "no-think" in output_dir.lower():
                    token_usage = usage_metadata.get("candidatesTokenCount")
                else:
                    token_usage = usage_metadata.get("thoughtsTokenCount")

                thinking = ""
                answer = ""
                final_answer = ""
                for part in parts:
                    if 'thought' in part:
                        thinking = part['text']
                    else:
                        answer = part['text']
                        output = parse_output(answer, subset=subset)
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


def save_results_openai(job_name, output_dir, gold_answers=None, subset=None):

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
                        output = parse_output(completion, subset=subset)
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

def save_results_together(job_name, output_dir, gold_answers=None, subset=None):
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
            output = {}
            confidence = ""
            confidence_score = ""
            if "gpt-oss" not in output_dir: # for non-reasoning models with final answer format

                final_answer_idx = completion.rfind("Final Answer:")
                if final_answer_idx > -1:
                    final_answer = completion[final_answer_idx:]
                    output = parse_output(final_answer.replace("*","").replace("<", "").replace(">", ""), subset=subset)

            else:
                
                reasoning = item['response']['body']['choices'][0]['message']['reasoning']
                if completion.strip():
                    output = parse_output(completion, subset=subset)
                else:
                    output = parse_output(completion, subset=subset)     
            
            if output:
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
        "--bench-root-dir",
        type=str,
        default="data/bench",
        help="Root directory of the benchmark datasets."
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
    job_name = args.job_name 
    output_dir = args.output_dir 

    if "replace_gold" in output_dir:
        position_abstain = "replace_gold"
    elif "additional" in output_dir:
        position_abstain = "additional"
    elif "first" in output_dir:
        position_abstain = "first"
    else:
        position_abstain = "last"

    if "afrimedqa" in output_dir:
        subset = "afrimedqa"
    elif "medqa_4opt" in output_dir:
        subset = "medqa_4opt"
    elif "medqa_5opt" in output_dir:
        subset = "medqa_5opt"
    elif "medmcqa" in output_dir:
        subset = "medmcqa"
    elif "medxpertqa-MM" in output_dir:
        subset = "medxpertqa-MM"
    elif "medxpertqa" in output_dir:
        subset = "medxpertqa"
    else:
        raise ValueError("Subset not found in output directory path.")
    
    if "life-threatening" in output_dir:
        question_type = "life-threatening"
    elif "safe" in output_dir:
        question_type = "safe"
    else:
        raise ValueError("Question type not found in output directory path.")

    benchmark = load_benchmark(input_dir=args.bench_root_dir, subset=subset, question_type=question_type, limit=None)
    gold_answers = get_gold_answers(benchmark=benchmark, output_dir=output_dir, position_abstain=position_abstain)

    if "gemini" in output_dir:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)
        #batch_job = client.batches.get(name=job_name)
        print("Processing results...")
        print("Susbet:", subset)
        save_results_gemini(job_name, output_dir, gold_answers=gold_answers, subset=subset)
        print("Done!")

    elif "together" in output_dir:
        client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
        print("Processing results...")
        save_results_together(job_name, output_dir, gold_answers=gold_answers, subset=subset)
        print("Done!")

    elif "openai" in output_dir:
        OPENAI_KEY = os.getenv("OPENAI_KEY")
        client = OpenAI(api_key=OPENAI_KEY)
        print("Processing results...")
        save_results_openai(job_name, output_dir, gold_answers=gold_answers, subset=subset)
        print("output_dir:", output_dir)
        print("Done!")

    else:
        print("Unknown model in output directory path.")

   