import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from datetime import datetime
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
    elif "medxpertqa" in subset:
        from datasets import load_dataset
        if "MM" in subset:
            benchmark = load_dataset('TsinghuaC3I/MedXpertQA', "MM", split="test")
        else:
            benchmark = load_dataset('TsinghuaC3I/MedXpertQA', "Text", split='test')
        if limit is not None:
            benchmark = benchmark.select(range(limit))
    elif "afrimedqa" in subset:
        from datasets import load_dataset
        from collections import Counter
        benchmark = load_dataset('afrimedqa/afrimedqa_v2')['train']
        # filter for mcq only questions
        benchmark = benchmark.filter(lambda x: x['question_type'] == 'mcq' and x['split'] == "test")
        print(f"MCQ data: {len(benchmark)}")
        # remove 3 opt questions
        benchmark = benchmark.filter(lambda x: dict(Counter(eval(x['answer_options']).values())).get("n/a", 0) < 2)
        print(f"MCQ data after 3-options questions removal: {len(benchmark)}")
        
        benchmark = benchmark.filter(lambda x: len(x['correct_answer'].split(",")) == 1)
        print(f"MCQ data after multiple-answer questions removal: {len(benchmark)}")
        benchmark = benchmark.rename_column("sample_id", "id")

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
            elif "medxpertqa" in subset:
                options = format_options(item['options'], subset="medxpertqa")
                prompt = PROMPT_TEMPLATE.replace("<QUESTION>", item['question'].split("Answer Choices:")[0].strip()).replace("<OPTIONS>", options)
            elif "afrimedqa" in subset:
                options = format_options(eval(item['answer_options']), subset="afrimedqa")
                prompt = PROMPT_TEMPLATE.replace("<QUESTION>", item['question_clean']).replace("<OPTIONS>", options)


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



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")
    
    parser.add_argument(
        "--subset",
        type=str,
        default="afrimedqa",
        choices=["medxpertqa", "medmcqa", "medqa", "medxpertqa-MM", "afrimedqa"],
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
   