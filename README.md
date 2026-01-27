# LLMs (Almost) Never Abstain Under Medical Uncertainty

This repository contains the code and data for our paper **"LLMs (Almost) Never Abstain Under Medical Uncertainty"**.

## Overview

Current medical AI benchmarks assume models should always commit to an answer. But in clinical practice, **abstaining is often the only safe action** when faced with uncertainty. We introduce **MedQAbstain**, a benchmark that evaluates whether LLMs can safely abstain from medical decisions—a critical but overlooked capability for clinical deployment.

### Key Finding

State-of-the-art LLMs systematically overcommit to clinical actions, **rarely abstaining even when the question itself is hidden**. This reveals a fundamental mismatch between LLM behavior and safe clinical decision-making.

## What is MedQAbstain?

MedQAbstain repurposes standard medical MCQA datasets to evaluate abstention as a safety-critical decision:

- ❌ **Removes the gold answer** from available options

- ✅ **Adds an explicit "I abstain" option** (e.g., "escalate to another physician")

- 🎯 **Makes abstention the only correct choice** since no safe action remains

- 📊 **Elicits self-reported confidence** to analyze calibration and decision-making

This shifts evaluation from **epistemic correctness** (knowledge) to **safety-critical decision-making under uncertainty**.

### Medical Abstention vs. Epistemic Abstention

| Type                     | Definition                                            | Example                                                                   |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| **Epistemic Abstention** | Not answering due to lack of knowledge                | "I don't know the capital of Uzbekistan"                                  |
| **Medical Abstention**   | Refraining from action because acting would be unsafe | "I cannot prescribe without complete patient history—refer to specialist" |

In safety-critical domains like medicine, this distinction is essential. A model can have partial information yet still need to abstain because the risk of acting outweighs the potential benefit.

---

## Typical Workflow

A typical usage workflow of the benchmark consists of the following steps:

1. [**Clone Repository**](#installation): Clone the repository and install dependencies.

2. [**Data Preparation**](#data):
   Either download the preprocessed datasets or regenerate them from scratch by following the steps in [**Data Preprocessing**](#data-preprocessing).

3. [**Environment Setup**](#environment-setup):
   Install dependencies and configure the required API keys.

4. [**Run Evaluation**](#quick-start):
   Evaluate models using local or API-based inference across different datasets and perturbation settings (e.g., masking, adversarial attacks, direct inference).

5. [**Compute Metrics**](#compute-metrics):
   Compute evaluation metrics, including **Abstention Rate**, **Expected Calibration Error (ECE)**, **Brier Score**, and **AUROC**, using the generated model outputs.

---

## Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/llm-medical-abstention-2D5E
cd llm-medical-abstention

# Install dependencies
pip install -r requirements.txt
```

---

## Data

The final datasets used for model evaluation are available for download as a ZIP archive at the following link: [[**Download Data**](https://drive.google.com/file/d/1KZAaMD-EbJlgNfddXLKYy3VkO2NU610P/view?usp=sharing)]

After extracting the archive, place the resulting `data/` directory in the **root of the project**.

The `data/` directory contains the following subfolders:

* **`bench/`**
  Contains one folder per dataset included in *MedQAbstain*, each further split into **life-threatening** and **safe** subsets.

* **`images/`**
  Contains image files for **MedXpertQA-MM**. These images are sourced directly from the original HuggingFace dataset:
  [[MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)]

* **`swap/`**
  Contains datasets used for the **option-swapping ablation study**, including **MedQA-4opt** and **MedXpertQA-Text**, where answer options have been randomly swapped across questions.

If you wish to reproduce the dataset from scratch, including threat classification, subsampling, and option swapping, please refer to Section **[Data Preprocessing](#data-preprocessing)**.

---


## Environment Setup

Before running any inference or preprocessing steps, configure the required API credentials.
Create a `.env` file in the root directory of the project and add the following environment variables:

```bash
GEMINI_API_KEY=<your_gemini_api_key_here>
HUGGINGFACE_TOKEN=<your_huggingface_token_here>
TOGETHER_API_KEY=<your_together_api_key_here>
OPENAI_API_KEY=<your_openai_api_key_here>
```

These variables are automatically loaded at runtime using `load_dotenv()`, so no manual exporting is required.

## Quick Start

We provide two inference modes supporting both **text-only** and **multimodal** evaluation:

* **Local inference** with open models via **vLLM**
* **API-based inference** using **OpenAI, Google, and Together AI**

---

### Configuration Parameters

| Parameter              | Description                        | Options / Default                                                                 |
| ---------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| `--subset`             | Dataset to evaluate                | `medxpertqa`, `medmcqa`, `medqa_4opt`, `medqa_5opt`, `afrimedqa`, `medxpertqa-MM` |
| `--model-name`         | Model to use for inference         | See **Supported Models**                                                          |
| `--input-dir`          | Directory containing input data    | Default: `data/bench`                                                             |
| `--output-dir`         | Directory for saving completions   | Default: `out/completions`                                                        |
| `--limit`              | Limit number of samples (optional) | Integer or empty for all                                                          |
| `--position-abstain`   | Position of abstention option      | `last`, `first`, `replace_gold`, `last_none`, `additional`                        |
| `--question-type`      | Type of questions to evaluate      | `life-threatening`, `safe`                                                        |
| `--mask-question`      | Mask question text                 | Flag                                                                              |
| `--swap-options`       | Swap answer options                | Flag                                                                              |
| `--multimodal`         | Enable multimodal evaluation       | Flag                                                                              |
| `--mask-image`         | Mask image input                   | Flag (multimodal only)                                                            |
| `--adversarial-attack` | Enable adversarial attack          | Flag                                                                              |
| `--mask-emotion`       | Mask emotional stimuli             | Flag                                                                              |
| `--direct-inference`   | Disable CoT, force direct answer   | Flag                                                                              |
| `--batch-size`         | Batch size for inference           | Integer (e.g., `48`)                                                              |

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
    --adversarial-attack
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
Multimodal inference on MedXpertQA
```bash
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
    --adversarial-attack
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
    --model-name gpt-5-mini \
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
    --adversarial-attack
```

**Direct inference without CoT reasoning**  
The model is prompted to answer directly with a letter instead of reasoning step by step:
```bash
python3 -m src.bench_api \
    --subset afrimedqa \
    --model-name meta-llama/Llama-3.3-70B-Instruct-Turbo \
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
    --model-name openai/gpt-oss-120b \
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
    --model-name gemini-2.5-flash \
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

**Mask both + Adversarial attack**  
All input sources are masked, but an adversarial sentence is added to push the model to answer anyway:
```bash
python3 -m src.bench_api \
    --subset medxpertqa-MM \
    --model-name gemini-2.5-flash \
    --multimodal \
    --mask-question \
    --mask-image \
    --adversarial-attack
```

---

## Compute Metrics

After running inference, evaluation metrics can be computed directly from the generated prediction files (`.jsonl`).
Metrics are aggregated across datasets and saved both as **numeric results** and **visualizations**.

Below is an example script showing how to compute metrics for a single model evaluated across multiple benchmarks.

---

### Example: Computing Metrics for a Model

The script below collects the output paths corresponding to different datasets and computes all evaluation metrics in a single run.

```bash
#!/bin/bash

# Input paths for model generations (one per dataset)
INPUT_PATH_MEDQA_5OPT="out/completions/gemini_api/gemini-2.5-flash/medqa_5opt/life-threatening/last/mask_question/2025-12-16_14-19-57/generations_medqa_5opt.jsonl"
INPUT_PATH_MEDQA_4OPT="out/completions/gemini_api/gemini-2.5-flash/medqa_4opt/life-threatening/last/mask_question/2025-12-16_11-48-40/generations_medqa_4opt.jsonl"
INPUT_PATH_MEDMCQA="out/completions/gemini_api/gemini-2.5-flash/medmcqa/life-threatening/last/mask_question/2025-12-16_14-20-26/generations_medmcqa.jsonl"
INPUT_PATH_MEDXPERTQA="out/completions/gemini_api/gemini-2.5-flash/medxpertqa/life-threatening/last/mask_question/2025-12-16_14-20-12/generations_medxpertqa.jsonl"
INPUT_PATH_AFRIMEDQA="out/completions/gemini_api/gemini-2.5-flash/afrimedqa/life-threatening/last/mask_question/2025-12-16_11-46-10/generations_afrimedqa.jsonl"
INPUT_PATH_MEDXPERTQA_MM=""

# Output directories
OUT_PLOTS_DIR="out/plots"
OUT_METRICS_DIR="out/metrics"

# Collect non-empty input paths
INPUT_PATHS=()

for p in \
  "$INPUT_PATH_MEDQA_5OPT" \
  "$INPUT_PATH_MEDQA_4OPT" \
  "$INPUT_PATH_MEDMCQA" \
  "$INPUT_PATH_MEDXPERTQA" \
  "$INPUT_PATH_MEDXPERTQA_MM" \
  "$INPUT_PATH_AFRIMEDQA"
do
  [[ -n "$p" ]] && INPUT_PATHS+=("$p")
done

# Run metric computation
python3 -m src.utils.compute_metrics_all \
  --input-paths "${INPUT_PATHS[@]}" \
  --out-plots-dir "$OUT_PLOTS_DIR" \
  --out-metrics-dir "$OUT_METRICS_DIR"
```

---

## Data Preprocessing

All data preprocessing scripts are located in the [`process_data/`](process_data/) directory.
The pipeline below is applied **per dataset subset** (e.g., MedQA, MedMCQA, AfriMed-QA, etc.).

---

### Step 1: Threat Classification (Life-threatening vs. Safe)

For each selected subset, we classify every question as **life-threatening** or **safe**.
This is done using **Gemini-2.5-Flash** as an automatic judge via the **Gemini Batch API**.

```bash
python3 process_data/classify_threat_data_api.py \
    --subset "afrimedqa" \
    --input-dir "data/bench" \
    --output-dir "out/classification"
```

---

### Step 2: Retrieve Classification Results

After the batch job completes, retrieve the classification results for the selected subset:

```bash
python3 process_data/results_threat_data_api.py \
    --output-dir "out/classification/gemini_api/gemini-2.5-flash/afrimedqa/2025-12-09_21-17-23"
    --job-name "batches/iy79im8i5mk2y7c7dsnqkiox39kzj5pwuawx"

```
---

### Step 3: Save the Processed Dataset Locally

The classified dataset is then saved to disk for downstream evaluation:

```bash
python3 process_data/save_dataset.py \
    --benchmark afrimedqa \
    --input-file "out/classification/gemini_api/gemini-2.5-flash/medxpertqa-MM/2025-12-05_11-04-09/classification_afrimedqa.jsonl"
```
---

### Step 4: Subsampling Safe Instances (MedMCQA & AfriMed-QA)

For **MedMCQA** and **AfriMed-QA**, the number of instances labeled as *safe* is substantially larger.
To keep experiments scalable, we subsample **1,000 safe instances per dataset**, using stratified sampling.

#### MedMCQA

Stratified by the `subject_name` field:

```bash
python3 process_data/filter_medmcqa.py
```

#### AfriMed-QA

Stratified by the `specialty` field:

```bash
python3 process_data/filter_afrimedqa.py
```

---

### Optional: Answer Option Swapping (Ablation Study)

For the option-swapping ablation, we randomly replace a group of answer options from one question with those of another.
This experiment is conducted **only for MedQA-4opt and MedXpertQA**.

```bash
python3 process_data/swap_options.py \
    --input-dir "data/bench" \
    --output-dir "data/swap" \
    --subset "medxpertqa" \
    --mode "group"
```

---