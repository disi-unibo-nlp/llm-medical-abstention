import os
import re
import argparse
import json
from datetime import datetime
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple, Optional
from dotenv import load_dotenv  
from PIL import Image
from huggingface_hub import login
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from src.utils.utils import parse_output, format_prompts, load_benchmark, get_gold_answers, setup_output_dir, save_inputs_prompts_to_txt
load_dotenv() 
login(token=os.environ.get("HUGGINGFACE_TOKEN"))


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    id_prompts: list[str]
    data_info: Optional[list[dict]] = None
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None
    image_data: list[Image.Image] = None


def run_phi(input_requests: list[str], modality=None) -> ModelRequestData:
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="auto",
        enforce_eager=False,
    )

    prompts, id_prompts, data_info = [], [], []
    
    for item in input_requests:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": item['prompt'].strip()}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[]
    ), tokenizer


def run_mediphi(input_requests: list[str], modality=None) -> ModelRequestData:
    
    model_name = "microsoft/MediPhi-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="auto",
        enforce_eager=False,
    )

    prompts, id_prompts, data_info = [], [], []
    
    for item in input_requests:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": item['prompt'].strip()}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[]
    ), tokenizer


def run_llama3(input_requests: list[str], modality=None) -> ModelRequestData:
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="auto",
        enforce_eager=False,
    )

    prompts, id_prompts, data_info = [], [], []
    
    for item in input_requests:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": item['prompt'].strip()}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[]
    ), tokenizer


def run_med42(input_requests: list[str], modality=None) -> ModelRequestData:
    
    model_name = "m42-health/Llama3-Med42-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="auto",
        enforce_eager=False,
    )

    prompts, id_prompts, data_info = [], [], []
    
    for item in input_requests:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": item['prompt'].strip()}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[]
    ), tokenizer


def run_gemma3(input_requests, multimodal=False, model_name="gemma3"):
    
    model_name = "google/gemma-3-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    if multimodal:
        image_urls = [el["data_info"]["images"] for el in input_requests]
    
    prompts, id_prompts, data_info = [], [], []
    
    for i, item in enumerate(input_requests):

        if multimodal:
            placeholders = [{"type": "image", "image": "data/images/medxpertqa/" + url} for url in image_urls[i]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *placeholders,
                        {"type": "text", "text": item['prompt'].strip()},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item['prompt'].strip()},
                    ],
                }
            ]


        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)
        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])
    
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[Image.open("data/images/medxpertqa/" + url) for image_list in image_urls for url in image_list] \
        if multimodal else None,
    ), tokenizer


def run_medgemma(input_requests, multimodal=False, model_name="medgemma"):
    
    model_name = "google/medgemma-4b-it" if "1.5" not in model_name else "google/medgemma-1.5-4b-it"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    if multimodal:
        image_urls = [el["data_info"]["images"] for el in input_requests]
    
    prompts, id_prompts, data_info = [], [], []
    
    for i, item in enumerate(input_requests):

        if multimodal:
            placeholders = [{"type": "image", "image": "data/images/medxpertqa/" + url} for url in image_urls[i]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *placeholders,
                        {"type": "text", "text": item['prompt'].strip()},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item['prompt'].strip()},
                    ],
                }
            ]


        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)
        id_prompts.append(item['id_prompt'])
        data_info.append(item['data_info'])

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        data_info=data_info,
        image_data=[Image.open("data/images/medxpertqa/" + url) for image_list in image_urls for url in image_list] \
        if multimodal else None,
    ), tokenizer


model_example_map = {
    "medgemma": run_medgemma,
    "medgemma-1.5": run_medgemma,
    "med42": run_med42,
    "gemma3": run_gemma3,
    "mediphi": run_mediphi,
    "llama3": run_llama3,
    "phi": run_phi
}


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():

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
        default="medgemma", 
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
        "--modality",
        type=str,
        default="image",
        choices=["image"],
        help="Modality of the input.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Set the batch size of completions.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4000,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal "
        "data) for each request.",
    )

    parser.add_argument(
        "--adversarial-attack",
        action="store_true",
        help="Adversarial attack by hiding question content (MedQA only)."
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

    return parser.parse_args()


def main(args):
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
    model_type = "instruct"

    if mask_image:
        multimodal = False

    if swap_options:
        question_type += "-swap"
    
    api_dir = "vllm"

    output_dir = setup_output_dir(output_dir, api_dir, model_name, subset, question_type, position_abstain, now_dir, mask_question, mask_image, adversarial_attack, mask_emotion, direct_inference)
    os.makedirs(output_dir, exist_ok=True)

    benchmark = load_benchmark(input_dir, subset, question_type, limit)
    gold_answers = get_gold_answers(benchmark, output_dir, position_abstain)

    if model_name not in model_example_map:
        raise ValueError(f"Model type {model_name} is not supported.")

    prompts = format_prompts(benchmark, subset, position_abstain=position_abstain, mask_question=mask_question, model_type=model_type, adversarial_attack=adversarial_attack, mask_emotion=mask_emotion,direct_inference=direct_inference)
    assert len(prompts) == len(benchmark)

    input_requests = []
    for i, item in enumerate(benchmark):

        if "afrimedqa" in subset:
            id_prompt = prompts[i][0].split("-")[-1]
        elif "medqa" in subset:
            id_prompt = prompts[i][0].split("-")[-1]
        elif "medxpertqa-MM" in subset:
            id_prompt = prompts[i][0].split("-")[-1]
            id_prompt = "MM-" + id_prompt
        elif "medxpertqa" in subset:
            id_prompt = prompts[i][0].split("-")[-1]
            id_prompt = "Text-" + id_prompt
        elif "medmcqa" in subset:
            id_prompt = prompts[i][0].split("-", 1)[-1]

        prompt = prompts[i][1]
        item['gold_answer'] = gold_answers[id_prompt]

        input_requests.append({
            "id_prompt": id_prompt,
            "prompt": prompt,
            "data_info": item
        })

    if "gemma" in model_name.lower():
        
        req_data, tokenizer = model_example_map[model_name](input_requests, multimodal, model_name=model_name)
        # Disable other modalities 
        default_limits = {"image": 6, "video": 0, "audio": 0}
        req_data.engine_args.limit_mm_per_prompt = default_limits 

        engine_args = asdict(req_data.engine_args) | {
            "seed": args.seed,
            "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
        }
    else: 
        req_data, tokenizer = model_example_map[model_name](input_requests, multimodal)
        engine_args = asdict(req_data.engine_args) | {
            "seed": args.seed
        }

    llm = LLM(**engine_args)

    prompts = req_data.prompts
    id_prompts = req_data.id_prompts
    data_info = req_data.data_info
    image_data = req_data.image_data if not mask_image else None
    
    sampling_params = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, stop_token_ids=req_data.stop_token_ids
    )

    
    if "gemma" in model_name.lower():
        inputs = [

            {
                "meta_data": {
                    "id_prompt": id_prompts[i],
                    "answer": data_info[i]['gold_answer']
                },
                "request": {   
                    "prompt": prompts[i],
                    "multi_modal_data": {"image": image_data[i] if image_data else []},
                }
            }
            for i in range(len(prompts))
        ]
    else:
        inputs = [
            {
                "meta_data": {
                    "id_prompt": id_prompts[i],
                    "answer": data_info[i]['gold_answer']
                },
                "request": prompts[i]
            }
            for i in range(len(prompts))
        ]

    print(f"Saving input prompts to {output_dir}/input_prompts.txt")
    save_inputs_prompts_to_txt(inputs, f"{output_dir}/input_prompts.txt", prompts_limit=10)
    print("Done.")

    # Batch inference
    batched_inputs = [inputs[i : i + args.batch_size] for i in range(0, len(inputs), args.batch_size)]

    for batch in tqdm(batched_inputs):
        ids = [el['meta_data']['id_prompt'] for el in batch]
        input_batch = [el["request"] for el in batch]
        gold_answers = [el['meta_data']['answer'] for el in batch]

        with time_counter(args.time_generate):
            outputs = llm.generate(
                input_batch,
                sampling_params=sampling_params,
                lora_request=None,
                use_tqdm=False
            )

        print("-" * 50)
        for id_out, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            reasoning = ""
            print("ID Question:", ids[id_out])
            print(generated_text)
            print("-" * 50)
            reasoning = generated_text.split("Final Answer:")[0].strip() if "Final Answer:" in generated_text else ""
            matches = list(re.finditer(r"final\s*answer\s*:", generated_text, re.IGNORECASE))
            if matches:
                generated_text = generated_text[matches[-1].start():]
            generated_text = generated_text.replace("*", "").replace("<", "").replace(">", "")
            output = parse_output(generated_text, subset=subset)

            final_answer = output['answer'] 
            confidence = output['confidence']
            confidence_score = output['confidence_score']
            
            result = {"id_question": ids[id_out], "dataset": subset, "gold_answer": gold_answers[id_out], "final_answer": final_answer,  "confidence": confidence, "confidence_score": confidence_score, "correct": gold_answers[id_out] == final_answer, "completion": generated_text, "thinking": reasoning, "thinking_length": len(tokenizer.encode(reasoning))}
           
            with open(f"{output_dir}/generations_{subset}.jsonl", "a") as f:
                json.dump(result, f)
                f.write("\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)