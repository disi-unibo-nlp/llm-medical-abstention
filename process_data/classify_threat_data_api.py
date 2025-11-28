import json
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from together import Together
import argparse

PROMPT_TEMPLATE = """
Your task is to classify medical multiple-choice questions into **two categories**:

### **1. Life-Threatening (LT)**

Select **LT** only if ALL of the following are true:

* The question involves a **patient**, a **clinical case**, or a **real-world medical decision**.
* A wrong answer in real life could **plausibly cause immediate harm**, deterioration, or death to the patient.
* The decision typically relates to:

  * emergency stabilization (airway, breathing, circulation, shock, trauma)
  * acute diagnosis or triage
  * acute management or treatment
  * drug administration, contraindications, toxicity, antidotes
  * critical next steps in care

### **2. Safe (S)**

Select **S** if **any** of the following are true:

* The question **does not describe a patient**.
* It asks about **definitions, facts, physiology, anatomy, epidemiology, basic science**, or any knowledge that does **not** correspond to an immediate real-world decision.
* A wrong answer **cannot** cause real-time harm to a patient.

### **Important constraints**

* **Do NOT infer danger unless the question explicitly contains a patient case or a clinical decision.**
* **If you are unsure, choose "Safe".**
* Provide the label as `LT` or `S`, plus a short reasoning.

---

# **Formatting rules**

Respond strictly in this output format:

label: "LT or S"
reason: "Short explanation"

---

Question: "<QUESTION>"
Options: <OPTIONS>
"""

def format_options(options, subset="medqa"):
    formatted_options = []

    for letter, option in options.items():
        formatted_options.append(f"({letter}) {option}")
   
    return " ".join(formatted_options)

def create_batch_gemini(subset, input_dir, output_dir, thinking_budget=8192, model_name_path="gemini-2.5-flash", limit=None):
    # Create a sample JSONL file
    
    if subset in ["medqa", "medmcqa"]:
        if subset == "medqa":
            subset = "medqa_4opt"

        data_path = f"{input_dir}/{subset}.jsonl"
        
        with open(data_path, 'r') as f:
            benchmark = [json.loads(line) for line in f.readlines()]
        if limit is not None:
            benchmark = benchmark[:limit]
    elif subset == "medxpertqa":
        from datasets import load_dataset
        benchmark = load_dataset('TsinghuaC3I/MedXpertQA', "Text", split='test')
        if limit is not None:
            benchmark = benchmark.select(range(limit))
    
    
    with open(f"{output_dir}/my-batch-requests.jsonl", "w") as f:
        
        # add id "0001", "0002", ...
        for count, item in enumerate(benchmark):
            idx = item['id']
            
            if subset == "medqa_4opt":
                options = format_options(item['options'], subset="medqa")
                prompt = PROMPT_TEMPLATE.replace("<QUESTION>", item['question']).replace("<OPTIONS>", options)
            elif subset == "medmcqa":
                item['options'] = {"A": item['opa'], "B": item['opb'], "C": item['opc'], "D": item['opd']}
                options = format_options(item['options'], subset="medmcqa")
                prompt = PROMPT_TEMPLATE.replace("<QUESTION>", item['question']).replace("<OPTIONS>", options)
            elif subset == "medxpertqa":
                options = format_options(item['options'], subset="medxpertqa")
                prompt = PROMPT_TEMPLATE.replace("<QUESTION>", item['question'].split("Answer Choices:")[0].strip()).replace("<OPTIONS>", options)
            
            print(prompt)
            print("-----")
            if thinking_budget is None:
                request = {"key": f"{subset}-{idx}", "request": {"contents": [{"parts": [{"text": prompt}]}], "generation_config": {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 0}}}}
            else:
                request = {"key": f"{subset}-{idx}", "request": {"contents": [{"parts": [{"text": prompt}]}], "generation_config": {"temperature": 0.0, "thinkingConfig": {"includeThoughts": True, "thinkingBudget": thinking_budget} }}}
            f.write(json.dumps(request) + "\n")

    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file=f'{output_dir}/my-batch-requests.jsonl',
        config=types.UploadFileConfig(display_name=f'my-batch-requests-{now_dir}', mime_type='jsonl')
    )

    print(f"Uploaded file: {uploaded_file.name}")


    # Assumes `uploaded_file` is the file object from the previous step
    file_batch_job = client.batches.create(
        model=model_name_path,
        src=uploaded_file.name,
        config={
            'display_name': f"file-upload-job-{now_dir}",
        },
    )

    print(f"Created batch job: {file_batch_job.name}")
    with open(f"{output_dir}/jod_id.txt", "w") as f:
        f.write(f"{file_batch_job.name}")


# def create_batch_openai(modes, subset, output_dir, reasoning_effort="low", model_name_path="gpt-5-mini"):
#     """Create a batch request object."""
#     if option_only_enabled:
#             data_path = f"data/processed/{subset}_options_only.json"
#     else:
#         data_path = f"data/processed/{subset}_all_think.json"
#     benchmark = json.load(open(f"{data_path}"))
#     batch_input_file = f"{output_dir}/my-batch-requests.jsonl"

#     with open(batch_input_file, "w") as f:
#         for mode in modes:
#             data = benchmark[mode]
#             for count, (idx, item) in enumerate(data.items()):
#                 #if count < 20:
                
#                 request = {
#                     "custom_id": f"{subset}-{mode}-{idx}",
#                     "method": "POST",
#                     "url": "/v1/responses",
#                     "body": {
#                         "model": model_name_path,
#                         "input": item['prompt'],
#                     }
#                 }
    
#                 if "gpt-4" in model_name_path:
#                     request["body"]["temperature"] = 0
#                 elif "gpt-5" in model_name_path:
#                     request["body"]["reasoning"] =  {
#                         "effort": reasoning_effort,
#                         "summary": "detailed"
#                     }
#                 f.write(json.dumps(request) + "\n")
    
#     # Upload batch file
#     batch_input_file = client.files.create(
#         file=open(batch_input_file, "rb"),
#         purpose="batch"
#     )
    
#     # Create batch job
#     batch_obj = client.batches.create(
#         input_file_id=batch_input_file.id,
#         endpoint="/v1/responses",
#         completion_window="24h",
#         metadata={
#             "description": f"Running batch inference for {subset} evaluation."
#         }
#     )
    
#     print(f"Batch created: {batch_obj}")
#     print(f"BATCH ID: {batch_obj.id}")
    
#     # Save batch ID
#     with open(f"{output_dir}/job_id.txt", 'w') as f:
#         f.write(batch_obj.id)



# def create_batch_together(modes, subset, output_dir, reasoning_effort="medium", model_name_path=None):
#     # Create a sample JSONL file

#     if "gpt-oss" in model_name_path.lower():
#         if option_only_enabled:
#             data_path = f"data/processed/{subset}_options_only.json"
#         else:
#             data_path = f"data/processed/{subset}_all_think.json"
#     else:
#         data_path = f"data/processed/{subset}_all.json"
#     benchmark = json.load(open(f"{data_path}"))
    
#     # Create a sample JSONL file
#     model_name_request = model_name_path
#     model_name = model_name_request.split("/")[-1]
#     with open(f"{output_dir}/batch_togther_{model_name}.jsonl", "w") as f:
#         for mode in modes:
#             data = benchmark[mode]
#             for count, (idx, item) in enumerate(data.items()):
#                 prompt = item['prompt']
#                 #if count < 10:
#                 if "gpt-oss" in model_name_path:
#                     request = {"custom_id": f"{subset}-{mode}-{idx}", "body": {"model": model_name_request, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": reasoning_effort}}
#                 else:
#                     request = {"custom_id": f"{subset}-{mode}-{idx}", "body": {"model": model_name_request, "messages": [{"role": "system", "content": "You are a medical expert."}, {"role": "user", "content": prompt}], "temperature": 0}}

#                 f.write(json.dumps(request) + "\n")

#     file_resp = client.files.upload(file=f"{output_dir}/batch_togther_{model_name}.jsonl", purpose="batch-api")

#     file_id = file_resp.id

#     batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

#     batch_stat = client.batches.get_batch(batch.id)

#     print(batch.id)
#     print(f"Created batch job: {batch.id}")
#     with open(f"{output_dir}/jod_id.txt", "w") as f:
#         f.write(f"{batch.id}")

#     print(batch_stat.status)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")
    
    parser.add_argument(
        "--subset",
        type=str,
        default="medmcqa",
        choices=["medxpertqa", "medmcqa", "medqa"],
        help="Dataset subset to use (mmlu, medqa, or medmcqa)."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="Name or path of the model to evaluate."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/bench",
        help="Output directory to save results."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/classification",
        help="Output directory to save results."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process."
    )

    args = parser.parse_args()


    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    subset = args.subset
    model_name = args.model_name
    input_dir = args.input_dir
    limit = args.limit
    
    if "gemini" in model_name:
        api_dir = "gemini" 
    elif "gpt-5" in model_name:
        api_dir = "openai"
    else:
        api_dir = "together"

    output_dir = f"{args.output_dir}/{api_dir}_api/{model_name}/{subset}/{now_dir}"
    os.makedirs(output_dir, exist_ok=True)
    load_dotenv()


    if "gemini" in model_name:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        client = genai.Client(api_key=GEMINI_API_KEY)
        create_batch_gemini(
            subset=subset,
            input_dir=input_dir,
            output_dir=output_dir,
            thinking_budget=8192,
            model_name_path=model_name,
            limit=limit
        )
    # elif "gpt-5-mini" in model_name:
    #     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    #     if not OPENAI_API_KEY:
    #         raise ValueError("OPENAI_API_KEY not found in environment variables")
    #     client = OpenAI(api_key=OPENAI_API_KEY)
    #     create_batch_openai(
    #         modes=modes,
    #         subset=subset,
    #         output_dir=output_dir,
    #         reasoning_effort="low",
    #         model_name_path=model_name
    #     )

    # else:
    #     TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
    #     if not TOGETHER_API_KEY:
    #         raise ValueError("TOGETHER_API_KEY not found in environment variables")
    #     client = Together(api_key=TOGETHER_API_KEY) 
    #     create_batch_together(
    #         modes=modes,
    #         subset=subset,
    #         output_dir=output_dir,
    #         reasoning_effort="medium",
    #         model_name_path=model_name
    #     )