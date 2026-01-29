import json
import re
import os
import argparse
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from together import Together
from src.utils.utils import format_prompts, setup_output_dir, load_benchmark
load_dotenv()

def create_batch_gemini(benchmark, subset, output_dir, thinking_budget=8192, model_name_path="gemini-2.5-flash", limit=None, question_type="life-threatening", position_abstain="last", mask_question=False, multimodal=False, mask_image=False, adversarial_attack=False, mask_emotion=False, direct_inference=False):
    # Create a sample JSONL file

    model_type = "instruct" if "no-think" in model_name_path.lower() else "reasoner"
    model_name_path = model_name_path.replace("-no-think","").strip() if "no-think" in model_name_path.lower() else model_name_path

    if not multimodal:
        json_file_path = f"{output_dir}/my-batch-requests.jsonl"
        with open(json_file_path, "w") as f:
            
            prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, model_type=model_type, mask_question=mask_question, adversarial_attack=adversarial_attack, mask_emotion=mask_emotion, direct_inference=direct_inference)
            for id_prompt, prompt in prompts:
                request = {"key": id_prompt, "request": {"contents": [{"parts": [{"text": prompt}]}], "generation_config": {"temperature": 0.0, "thinkingConfig": {"includeThoughts": True, "thinkingBudget": thinking_budget} }}}
                
                if model_type != "reasoner":
                    request['request']['generation_config']['max_output_tokens'] = 8192

                f.write(json.dumps(request) + "\n")
    else:
        
        images = [item['images'] for item in benchmark]

        prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, model_type=model_type, mask_question=mask_question, adversarial_attack=adversarial_attack, mask_emotion=mask_emotion, direct_inference=direct_inference)
        assert len(images) == len(prompts)
        requests_data = []
        for i, img_paths in enumerate(images): 
            content_parts = []
            content_parts.append({"text": prompts[i][1]})
            
            if not mask_image:
                for image_path in img_paths:
                    image_path = "data/images/medxpertqa/" + image_path
                    print(f"Uploading image file: {image_path}")
                    image_file = client.files.upload(
                        file=image_path,
                    )
                    content_parts.append({"file_data": {"file_uri": image_file.uri, "mime_type": image_file.mime_type}})
                    print(f"Uploaded image file: {image_file.name} with MIME type: {image_file.mime_type}")
                    # with open(image_path, 'rb') as f:
                    #     image_bytes = f.read()
                    # content_parts.append({"file_data": types.Part.from_bytes(
                    #     data=image_bytes,
                    #     mime_type='image/jpeg',
                    # )})
    
            requests_data.append(
                #  request: multi-modal prompt with text and an image reference
                {
                    "key": prompts[i][0],
                    "request": {
                        "contents": [{
                            "parts": content_parts
                        }]
                    }
                }
            )


        json_file_path = f'{output_dir}/batch_requests_with_image.json'

        print(f"\nCreating JSONL file: {json_file_path}")
        with open(json_file_path, 'w') as f:
            for req in requests_data:
                f.write(json.dumps(req) + '\n')


    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file=json_file_path,
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


def create_batch_openai(benchmark, subset, output_dir, reasoning_effort="medium", model_name_path="gemini-2.5-flash", limit=None, question_type="life-threatening", position_abstain="last", mask_question=False, multimodal=False, mask_image=False, adversarial_attack=False):
    """Create a batch request object."""
    import base64
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    batch_input_file = f"{output_dir}/my-batch-requests.jsonl"
    if mask_image:
        multimodal = False

    if not multimodal:
        with open(batch_input_file, "w") as f:

            prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, mask_question=mask_question, adversarial_attack=adversarial_attack)
            limit = len(prompts) if limit is None else limit
            for id_prompt, prompt in prompts[:limit]:
                
                request = {
                    "custom_id": f"{id_prompt}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name_path,
                        "input": prompt,
                    }
                }

                if "gpt-4" in model_name_path:
                    request["body"]["temperature"] = 0
                elif "gpt-5" in model_name_path:
                    request["body"]["reasoning"] =  {
                        "effort": reasoning_effort,
                        "summary": "detailed"
                    }
                f.write(json.dumps(request) + "\n")
    else: # multimodal
        image_paths = [item['images'] for item in benchmark]
        prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, mask_question=mask_question, adversarial_attack=adversarial_attack)
        assert len(image_paths) == len(prompts)
        limit = len(prompts) if limit is None else limit
        with open(batch_input_file, "w") as f:
            for i, (id_prompt, prompt) in enumerate(prompts[:limit]):
                
                input_content = [{"type": "input_text", "text": prompt}]
                for img_path in image_paths[i]:
                    base64_image = encode_image("data/images/medxpertqa/" + img_path)
                    input_content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    })

                request = {
                    "custom_id": f"{id_prompt}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name_path,
                        "input": [{
                            "role": "user",
                            "content": input_content
                        }]
                    }
                }

                if "gpt-4" in model_name_path:
                    request["body"]["temperature"] = 0
                elif "gpt-5" in model_name_path:
                    request["body"]["reasoning"] =  {
                        "effort": reasoning_effort,
                        "summary": "detailed"
                    }
                f.write(json.dumps(request) + "\n")
        # Upload batch file
    batch_input_file = client.files.create(
        file=open(batch_input_file, "rb"),
        purpose="batch"
    )
    
    # Create batch job
    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": f"Running batch inference for {subset} evaluation."
        }
    )
    
    print(f"Batch created: {batch_obj}")
    print(f"BATCH ID: {batch_obj.id}")
    
    # Save batch ID
    with open(f"{output_dir}/job_id.txt", 'w') as f:
        f.write(batch_obj.id)



def create_batch_together(benchmark, subset, output_dir, reasoning_effort="medium", model_name_path=None, limit=None, question_type="life-threatening", position_abstain="last", mask_question=False, adversarial_attack=False, mask_emotion=False, direct_inference=False):
    # Create a sample JSONL file

    # Create a sample JSONL file
    model_name_request = model_name_path
    model_name = model_name_request.split("/")[-1]

    model_type = "instruct" if "instruct" in model_name.lower() else "reasoner"
    
    prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, model_type=model_type, mask_question=mask_question, adversarial_attack=adversarial_attack, mask_emotion=mask_emotion, direct_inference=direct_inference)

    with open(f"{output_dir}/batch_togther_{model_name}.jsonl", "w") as f:
        for id_prompt, prompt in prompts:
            if "gpt-oss" in model_name_path:
                request = {"custom_id": id_prompt, "body": {"model": model_name_request, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": reasoning_effort, "temperature": 0}}
            else:
                request = {"custom_id": id_prompt, "body": {"model": model_name_request, "messages": [{"role": "system", "content": "You are a medical expert."}, {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 4096}}

            f.write(json.dumps(request) + "\n")

    file_resp = client.files.upload(file=f"{output_dir}/batch_togther_{model_name}.jsonl", purpose="batch-api")

    file_id = file_resp.id

    batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

    batch_stat = client.batches.get_batch(batch.id)

    print(batch.id)
    print(f"Created batch job: {batch.id}")
    with open(f"{output_dir}/jod_id.txt", "w") as f:
        f.write(f"{batch.id}")

    print(batch_stat.status)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")
    
    parser.add_argument(
        "--subset",
        type=str,
        default="medqa_5opt",
        choices=["medxpertqa", "medmcqa", "medqa_4opt", "medqa_5opt", "afrimedqa", "medxpertqa-MM"],
        help="Dataset subset to use (mmlu, medqa, or medmcqa)."
    )

    parser.add_argument(
        "--question-type",
        type=str,
        default="life-threatening",
        choices=["life-threatening", "safe"],
        help="Type of questions to evaluate."
    )

    parser.add_argument(
        "--position-abstain",
        type=str,
        default="last",
        choices=["last", "first", "replace_gold", "last_none", "additional"],
        help="Position of the 'I abstain' option."
    )


    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-120b", # openai/gpt-oss-120b | "gemini-2.5-flash" | "gemini-2.5-flash-no-think"
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
        default="out/completions",
        help="Output directory to save results."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process."
    )

    parser.add_argument(
        "--mask-question",
        action="store_true",
        help="Mask the question when enabled."
    )

    parser.add_argument(
        "--mask-image",
        action="store_true",
        help="Mask the image(s) when enabled."
    )

    parser.add_argument(
        "--swap-options",
        action="store_true",
        help="Mask the question when enabled."
    )

    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Multimodal evaluation when enabled."
    )

    parser.add_argument(
        "--adversarial-attack",
        action="store_true",
        help="Adversarial sentence attack when enabled."
    )

    parser.add_argument(
        "--mask-emotion",
        action="store_true",
        help="Mask emotional sentence in the prompt when enabled."
    )

    parser.add_argument(
        "--direct-inference",
        action="store_true",
        help="Direct inference without any CoT reasoning (non-reasoner models only)."
    )

    parser.add_argument(
        "--low-effort",
        action="store_true",
        help="Low effort reasoning (for OpenAI and Together models only)."
    )

    args = parser.parse_args()


    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    subset = args.subset
    model_name = args.model_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    limit = args.limit
    position_abstain = args.position_abstain
    question_type = args.question_type
    mask_question = args.mask_question
    swap_options = args.swap_options
    multimodal = args.multimodal
    mask_image = args.mask_image
    adversarial_attack = args.adversarial_attack
    mask_emotion = args.mask_emotion
    direct_inference = args.direct_inference
    low_effort = args.low_effort

    if swap_options:
        question_type += "-swap"
    
    if "gemini" in model_name:
        api_dir = "gemini_api" 
    elif "gpt-5" in model_name:
        api_dir = "openai_api"
    else:
        api_dir = "together_api"

    output_dir = setup_output_dir(
        output_dir=output_dir,
        api_dir=api_dir,
        model_name=model_name,
        subset=subset,
        question_type=question_type,
        position_abstain=position_abstain,
        now_dir=now_dir,
        mask_question=mask_question,
        mask_image=mask_image,
        adversarial_attack=adversarial_attack,
        mask_emotion=mask_emotion,
        direct_inference=direct_inference,
        low_effort=low_effort
    )

    os.makedirs(output_dir, exist_ok=True)
    benchmark = load_benchmark(input_dir=input_dir, subset=subset, question_type=question_type, limit=limit)

    if "no-think" in model_name.lower():
        thinking_budget = 0
    elif low_effort:
        thinking_budget = 1024
    else:
        thinking_budget = 8192
    
    if "gemini" in model_name:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        client = genai.Client(api_key=GEMINI_API_KEY)
        create_batch_gemini(
            benchmark=benchmark,
            subset=subset,
            output_dir=output_dir,
            thinking_budget=thinking_budget,
            model_name_path=model_name,
            limit=limit,
            question_type=question_type,
            position_abstain=position_abstain,
            mask_question=mask_question,
            multimodal=multimodal,
            mask_image=mask_image,
            adversarial_attack=adversarial_attack,
            mask_emotion=mask_emotion, 
            direct_inference=direct_inference
        )
    elif "gpt-5-mini" in model_name:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=OPENAI_API_KEY)
        create_batch_openai(
            benchmark=benchmark,
            subset=subset,
            output_dir=output_dir,
            reasoning_effort="medium" if not low_effort else "low",
            model_name_path=model_name,
            limit=limit,
            question_type=question_type,
            position_abstain=position_abstain,
            mask_question=mask_question,
            multimodal=multimodal,
            mask_image=mask_image,
            adversarial_attack=adversarial_attack
        )

    else:
        TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        client = Together(api_key=TOGETHER_API_KEY) 
        create_batch_together(
            benchmark=benchmark,
            subset=subset,
            output_dir=output_dir,
            reasoning_effort="medium" if not low_effort else "low",
            model_name_path=model_name,
            limit=limit,
            question_type=question_type,
            position_abstain=position_abstain,
            mask_question=mask_question,
            adversarial_attack=adversarial_attack,
            mask_emotion=mask_emotion,
            direct_inference=direct_inference
        )