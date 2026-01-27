# LLMs (Almost) Never Abstain Under Medical Uncertainty

This repository contains the code and data for our paper **"LLMs (Almost) Never Abstain Under Medical Uncertainty"**.

## Overview

Current medical AI benchmarks assume models should always commit to an answer. But in clinical practice, **abstaining is often the only safe action** when faced with uncertainty. We introduce **MedQAbstain**, a benchmark that evaluates whether LLMs can safely abstain from medical decisions—a critical but overlooked capability for clinical deployment.

### Key Finding

State-of-the-art LLMs systematically overcommit to clinical actions, **rarely abstaining even when the question itself is hidden**. This reveals a fundamental mismatch between LLM behavior and safe clinical decision-making.

## What is MedQAbstain?

MedQAbstain repurposes standard medical MCQA datasets to evaluate abstention as a safety-critical decision:

- ❌ **Removes the gold answer** from available options

- ✅ **Adds an explicit "I abstain" option** (e.g., "refer to specialist", "escalate to senior physician")

- 🎯 **Makes abstention the only correct choice** since no safe action remains

- 📊 **Elicits self-reported confidence** to analyze calibration and decision-making

This shifts evaluation from **epistemic correctness** (knowledge) to **safety-critical decision-making under uncertainty**.

### Medical Abstention vs. Epistemic Abstention

| Type                     | Definition                                            | Example                                                                   |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| **Epistemic Abstention** | Not answering due to lack of knowledge                | "I don't know the capital of Uzbekistan"                                  |
| **Medical Abstention**   | Refraining from action because acting would be unsafe | "I cannot prescribe without complete patient history—refer to specialist" |

In safety-critical domains like medicine, this distinction is essential. A model can have partial information yet still need to abstain because the risk of acting outweighs the potential benefit.

## Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/llm-medical-abstention-2D5E
cd llm-medical-abstention

# Install dependencies

pip install -r requirements.txt
```

## Quick Start

We provide two inference modes supporting both **text-only** and **multimodal** evaluation:

* **Local inference** with open models via **vLLM**
* **API-based inference** using **OpenAI, Google, and Together AI**

---

### Configuration Parameters

| Parameter            | Description                        | Options / Default                                                                 |
| -------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| `--subset`           | Dataset to evaluate                | `medxpertqa`, `medmcqa`, `medqa_4opt`, `medqa_5opt`, `afrimedqa`, `medxpertqa-MM` |
| `--model-name`       | Model to use for inference         | See **Supported Models**                                                          |
| `--input-dir`        | Directory containing input data    | Default: `data/bench`                                                             |
| `--output-dir`       | Directory for saving completions   | Default: `out/completions`                                                        |
| `--limit`            | Limit number of samples (optional) | Integer or empty for all                                                          |
| `--position-abstain` | Position of abstention option      | `last`, `first`, `replace_gold`, `last_none`, `additional`                        |
| `--question-type`    | Type of questions to evaluate      | `life-threatening`, `safe`                                                        |
| `--mask-question`    | Mask question text                 | Flag                                                                              |
| `--swap-options`     | Swap answer options                | Flag                                                                              |
| `--multimodal`       | Enable multimodal evaluation       | Flag                                                                              |
| `--mask-image`       | Mask image input                   | Flag (multimodal only)                                                            |
| `--adversial-attack` | Enable adversarial attack          | Flag                                                                              |
| `--mask-emotion`     | Mask emotional stimuli             | Flag                                                                              |
| `--direct-inference` | Disable CoT, force direct answer   | Flag                                                                              |
| `--batch-size`       | Batch size for inference           | Integer (e.g., `48`)                                                              |

---

### Supported Models

| Model            | Access | Identifier                                | Reference URL                                                                   |
| ---------------- | ------ | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Gemini-2.5-Flash | API    | `gemini-2.5-flash`                        | [Documentation](https://ai.google.dev/gemini-api/docs/models)                   |
| GPT-5-mini       | API    | `gpt-5-mini`                              | [Documentation](https://platform.openai.com/docs/models/gpt-5-mini)             |
| GPT-OSS-120B     | API    | `openai/gpt-oss-120b`                     | [Together AI](https://www.together.ai/models/gpt-oss-120b)                      |
| GPT-OSS-20B      | API    | `openai/gpt-oss-20b`                      | [Together AI](https://www.together.ai/models/gpt-oss-20b)                       |
| LLaMA-3.3-70B    | API    | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | [Together AI](https://www.together.ai/models/llama-3-3-70b)                     |
| Qwen3-235B       | API    | `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` | [Together AI](https://www.together.ai/models/qwen3-235b-a22b-instruct-2507-fp8) |
| LLaMA-3-8B       | Local  | `llama3`                                  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)       |
| LLaMA3-Med42-8B  | Local  | `med42`                                   | [HuggingFace](https://huggingface.co/m42-health/Llama3-Med42-8B)                |
| Gemma-3-4B       | Local  | `gemma3`                                  | [HuggingFace](https://huggingface.co/google/gemma-3-4b-it)                      |
| MedGemma-4B      | Local  | `medgemma`                                | [HuggingFace](https://huggingface.co/google/medgemma-4b-it)                     |
| Phi-3.5-mini     | Local  | `Phi-3.5-mini`                            | [HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)           |
| MediPhi-3.8B     | Local  | `mediphi`                                 | [HuggingFace](https://huggingface.co/microsoft/MediPhi-Instruct)                |

---          

### 1. Local Inference with Open Models (vLLM)

#### Text-Only Inference Example
Basic text-only inference on MedQA dataset
```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m src.bench_vllm \
    --subset medqa_4opt \
    --question-type life-threatening \
    --position-abstain last \
    --model-name llama3 \
    --input-dir data/bench \
    --output-dir out/completions \
    --batch-size 48
```

#### Text-Only Common Use Cases

**Evaluate with limited samples**  
Use only 100 samples for quick debugging:
```bash
python3 -m src.bench_vllm \
    --subset medqa_4opt \
    --model-name med42 \
    --limit 100
```

**Mask question (Trivial Abstention mode)**  
The input question is masked to evaluate the model under Trivial Abstention conditions:
```bash
python3 -m src.bench_vllm \
    --subset medmcqa \
    --model-name llama3 \
    --mask-question
```

**Mask question + Adversarial attack**  
The input question is masked, but an adversarial sentence is added to push the model to answer anyway, testing its robustness:
```bash
python3 -m src.bench_vllm \
    --subset medmcqa \
    --model-name llama3 \
    --mask-question \
    --adversial-attack
```

**Direct inference without CoT reasoning**  
The model is prompted to answer directly with a letter instead of reasoning step by step:
```bash
python3 -m src.bench_vllm \
    --subset afrimedqa \
    --model-name medgemma \
    --direct-inference
```

**Mask emotion**  
Remove the emotional stimuli sentence (e.g., "if you answer correctly..., if you answer wrong...") from the beginning of the prompt:
```bash
python3 -m src.bench_vllm \
    --subset afrimedqa \
    --model-name medgemma \
    --mask-emotion
```

---

#### Multimodal Inference Example
```bash
# Multimodal inference on MedXpertQA
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m src.bench_vllm \
    --subset medxpertqa-MM \
    --question-type life-threatening \
    --position-abstain last \
    --model-name medgemma \
    --input-dir data/bench \
    --output-dir out/completions \
    --multimodal
```

#### Multimodal Common Use Cases

**Mask question only**  
Mask the input question while keeping the image visible:
```bash
python3 -m src.bench_vllm \
    --subset medxpertqa-MM \
    --model-name medgemma \
    --multimodal \
    --mask-question
```

**Mask image only**  
Mask the input image while keeping the question visible:
```bash
python3 -m src.bench_vllm \
    --subset medxpertqa-MM \
    --model-name medgemma \
    --multimodal \
    --mask-image
```

**Mask both question and image**  
Remove both input sources (expected to trigger easy abstention by the model):
```bash
python3 -m src.bench_vllm \
    --subset medxpertqa-MM \
    --model-name medgemma \
    --multimodal \
    --mask-question \
    --mask-image
```

**Mask both + Adversarial attack**  
All input sources are masked, but an adversarial sentence is added to push the model to answer anyway:
```bash
python3 -m src.bench_vllm \
    --subset medxpertqa-MM \
    --model-name medgemma \
    --multimodal \
    --mask-question \
    --mask-image \
    --adversial-attack
```

### 2. API-based Inference (OpenAI, Google, Together AI)

> **Note:** API-based inference requires valid API keys for the respective services (OpenAI, Google AI, Together AI). Make sure to set up your API keys in environment variables or configuration files before running these commands.

#### Text-Only Inference Example
Basic text-only inference on MedQA dataset using API models

```bash
python3 -m src.bench_api \
    --subset medqa_4opt \
    --question-type life-threatening \
    --position-abstain last \
    --model-name gemini-2.5-flash \
    --input-dir data/bench \
    --output-dir out/completions
```

#### Text-Only Common Use Cases

**Evaluate with limited samples**  
Use only 100 samples for quick debugging:
```bash
python3 -m src.bench_api \
    --subset medqa_4opt \
    --model-name openai/gpt-oss-20b \
    --limit 100
```

**Mask question (Trivial Abstention mode)**  
The input question is masked to evaluate the model under Trivial Abstention conditions:
```bash
python3 -m src.bench_api \
    --subset medmcqa \
    --model-name Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --mask-question
```

**Mask question + Adversarial attack**  
The input question is masked, but an adversarial sentence is added to push the model to answer anyway, testing its robustness:
```bash
python3 -m src.bench_api \
    --subset medmcqa \
    --model-name gemini-2.5-flash \
    --mask-question \
    --adversial-attack
```

**Direct inference without CoT reasoning**  
The model is prompted to answer directly with a letter instead of reasoning step by step:
```bash
python3 -m src.bench_api \
    --subset afrimedqa \
    --model-name gpt-5-mini \
    --direct-inference
```

**Mask emotion**  
Remove the emotional stimuli sentence from the beginning of the prompt:
```bash
python3 -m src.bench_api \
    --subset afrimedqa \
    --model-name openai/gpt-oss-120b \
    --mask-emotion
```

**Low effort mode**  
Enable low effort responses to test model behavior under minimal engagement:
```bash
python3 -m src.bench_api \
    --subset medqa_4opt \
    --model-name meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --low-effort
```

---

#### Multimodal Inference Example
Multimodal inference on MedXpertQA using API models
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --question-type life-threatening \
    --position-abstain last \
    --model-name gpt-5-mini \
    --input-dir data/bench \
    --output-dir out/completions \
    --multimodal
```

#### Multimodal Common Use Cases

**Mask question only**  
Mask the input question while keeping the image visible:
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --model-name gpt-5-mini \
    --multimodal \
    --mask-question
```

**Mask image only**  
Mask the input image while keeping the question visible:
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --model-name gemini-2.5-flash \
    --multimodal \
    --mask-image
```

**Mask both question and image**  
Remove both input sources (expected to trigger easy abstention by the model):
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --model-name gpt-5-mini \
    --multimodal \
    --mask-question \
    --mask-image
```

**Evaluate with limited samples**  
Process only 20 samples for quick testing:
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --model-name gemini-2.5-flash \
    --multimodal \
    --limit 20
```