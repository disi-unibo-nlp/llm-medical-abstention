import json
import argparse
from google import genai
from google.genai import types
# from openai import OpenAI
# from together import Together
import time
from dotenv import load_dotenv
import os
load_dotenv()


import re

def parse_output(text):
    """
    Parses an output of the form:
    
    label: "LT or S"
    reason: "Short explanation"
    
    Returns a dict: {"label": ..., "reason": ...}
    """
    pattern = r'label:\s*"([^"]+)"\s*\s*reason:\s*"([^"]+)"'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        raise ValueError("Invalid format")

    return {
        "label": match.group(1),
        "reason": match.group(2)
    }

def save_results_gemini(job_name, output_dir):

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
                parts = item['response']['candidates'][0]['content']['parts'] if 'candidates' in item['response'] and 'content' in item['response']['candidates'][0] else []
                thinking = ""
                answer = ""
                final_answer = ""
                for part in parts:
                    if 'thought' in part:
                        thinking = part['text']
                    else:
                        answer = part['text']
                        final_answer = parse_output(answer)
                
                with open(f"{output_dir}/classification_{subset}.jsonl", "a") as f:
                    json.dump({"id_question": id_item, "dataset": subset, "final_answer": final_answer, "completion": answer, "thinking": thinking}, f)
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


# def save_results_openai(job_name, output_dir):

#     #print(client.batches.retrieve(args.batch_id))
#     response = client.batches.retrieve(job_name)
#     is_safe = False
#     if response.status =='completed':
#         print("INFERENCE COMPLETED!")
#         print(response)

#         if response.error_file_id:
#             print("ERROR FILE ID: ", response.error_file_id)
#             file_response = client.files.content(response.error_file_id)
#             for line in file_response.text.splitlines():
#                 print(line)
#         elif response.output_file_id:
#             is_safe = True
#             print("OUTPUT FILE ID: ", response.output_file_id)
#             file_response = client.files.content(response.output_file_id)
#             out_file_id = response.output_file_id
        
#             for line in file_response.text.splitlines():
#                 print(line)
#             print("Saving results...")
#             with open(f'{output_dir}/raw_completions.jsonl', 'w') as f:
#                 for line in file_response.text.splitlines():
#                     json.dump(json.loads(line), f, ensure_ascii=False)
#                     f.write('\n')
#             print("Done!")

#         if is_safe:
#             print("Parsing results...")
#             with open(f'{output_dir}/raw_completions.jsonl', 'r') as f:
#                 completions = [json.loads(line) for line in f.readlines()]
            
            
#             for item in completions:

#                 key_splits = item['custom_id'].split("-", 2)
#                 dataset_item = key_splits[0]
#                 mode_item = key_splits[1]
#                 if mode_item not in dataset_per_mode:
#                     continue
#                 id_item = key_splits[2]
#                 gold_answer = dataset_per_mode[mode_item][id_item]['answer'] if mode_item != "yes_no_maybe" else "Yes"
#                 usage_info = item['response']['body']['usage']['output_tokens_details']['reasoning_tokens']
#                 #result = json.loads(line)
#                 output = item['response']['body']['output']
#                 final_answer = ""
#                 thinking = ""
#                 for out in output:
#                     if out['type'] == "reasoning":
#                         thinking = out['summary'][0]['text'] if out['summary'] else ""
#                     elif out['type'] == "message":
#                         completion = out['content'][0]['text'] if out['content'] else ""
#                         final_answer = parse_final_answer(completion, mode_item, model_name="openai")
#                 print(f'{output_dir}/generations_{mode_item}.jsonl')
#                 with open(f'{output_dir}/generations_{mode_item}.jsonl', 'a') as f:
#                     json.dump({"id_question": id_item, "dataset": dataset_item, "mode": mode_item, "gold_answer": gold_answer, "final_answer": final_answer, "completion": completion, "thinking": thinking, "thinking_length": usage_info}, f)
#                     f.write("\n")
#             print("Done!")

#     else:
#         print("BATCH STILL PROCESSING...")
#         print(f"STATUS: {response.status}")

# def save_results_together(job_name, output_dir):
#     batch_stat = client.batches.get_batch(job_name)

#     print(batch_stat.status)
#     # # Get the batch status to find output_file_id
#     batch = client.batches.get_batch(job_name)
#     output_file = output_dir + "/raw_completions.jsonl"

#     if batch.status == "COMPLETED":
       
#         # Download the output file
#         client.files.retrieve_content(
#             id=batch_stat.output_file_id,
#             output=output_file,
#         )

#         with open(output_file) as f:
#             completions = [json.loads(line) for line in f.readlines()]
        
#         for item in completions:
#             parts = item['custom_id'].split("-", 2)
#             dataset = parts[0]
#             mode_item = parts[1]

#             if mode_item not in modes:
#                 continue

#             id = parts[2]
#             gold_answer = dataset_per_mode[mode_item][id]['answer'] if mode_item != "yes_no_maybe" else "Yes"
#             completion = item['response']['body']['choices'][0]['message']['content']
#             completion_length = item['response']['body']['usage']['completion_tokens']
#             final_answer = ""
#             reasoning = ""
#             if "gpt-oss" not in output_dir:
#                 #modes = ["incorrect", "none_of_the_provided", "options_only", "yes_no_maybe", "roman_numeral", "fixed_pos", "no_symbols"]

#                 final_answer_idx = completion.rfind("Final Answer:")
#                 if final_answer_idx > 0:
#                     final_answer = completion[final_answer_idx:]
#                     #reasoning = completion[:final_answer_idx]
#                     final_answer = parse_final_answer(final_answer.replace("*",""), mode_item, model_name="llama3")

#             else:
                
#                 reasoning = item['response']['body']['choices'][0]['message']['reasoning']
#                 if completion.strip():
                    
#                     if mode_item != "open":
#                         final_answer = parse_final_answer(completion, mode=mode_item, model_name="gpt-oss-120b")
#                     else:
#                         final_answer_idx = completion.rfind("Final Answer:")
#                         if final_answer_idx >= 0:
#                             final_answer = completion[final_answer_idx+len("Final Answer:"):].strip()
#                 else:
#                     if mode_item != "open":
#                         final_answer = parse_final_answer(completion, mode=mode_item, model_name="gpt-oss-120b")
#                     else:
#                         final_answer_idx = reasoning.rfind("Final Answer:")
#                         if final_answer_idx >= 0:
#                             final_answer = reasoning[final_answer_idx+len("Final Answer:"):].strip()
                
#             with open(output_dir + f"/generations_{mode_item}.jsonl", "a") as f: 
#                 out_dict = {"id_question": id, "mode": mode_item, "gold_answer": gold_answer, "final_answer": final_answer,  "correct": str(gold_answer).replace(".","") == str(final_answer).replace(".",""), "completion": completion, "thinking": reasoning, "thinking_length": completion_length }
#                 json.dump(out_dict, f)
#                 f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Retrieve and save API batch job results.',
    )  

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/classification/gemini_api/gemini-2.5-flash/medxpertqa-MM/2025-12-05_11-04-09",
        help="Output directory to save results."
    )

    parser.add_argument('--job-name', type=str, default="batches/cyd5ab5flc377dq0hiogv2u02jdlik3edav0",
                      help='batch job id to retrieve results from')
    
    args = parser.parse_args()

    
    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = args.job_name #batches/tdw033ksy8zwa7e8spuj30075my7031b041i" #"batches/jdsea2vgjodl3ftrdvxgpp3f478wwteg01ld" #"batches/taxgnuxeblk3sq3hkgxc6pmskwbaoqtx74kt"  # (e.g. 'batches/your-batch-id')
    output_dir = args.output_dir #"out/legal/completions/gemini_api/gemini-2.5-flash/professional_law/2025-10-18_00-19-13"

    if "medqa" in output_dir:
        subset = "medqa"
    elif "medmcqa" in output_dir:
        subset = "medmcqa"
    elif "medxpertqa" in output_dir:
        subset = "medxpertqa"
    else:
        raise ValueError("Subset not found in output directory path.")
    
    if "gemini" in output_dir:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)
        #batch_job = client.batches.get(name=job_name)
        print("Processing results...")
        save_results_gemini(job_name, output_dir)
        print("Done!")

    # elif "together" in output_dir:
    #     client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
    #     print("Processing results...")
    #     save_results_together(job_name, output_dir)
    #     print("Done!")

    # elif "openai" in output_dir:
    #     OPENAI_KEY = os.getenv("OPENAI_KEY")
    #     client = OpenAI(api_key=OPENAI_KEY)
    #     print("Processing results...")
    #     save_results_openai(job_name, output_dir)
    #     print("output_dir:", output_dir)
    #     print("Done!")

    # else:
    #     print("Unknown model in output directory path.")

   