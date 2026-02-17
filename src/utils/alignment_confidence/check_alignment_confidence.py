import json
import os
from dotenv import load_dotenv
load_dotenv()
from together import Together


alignment_prompt_template = """
You are evaluating whether a language model’s stated confidence level
is aligned with the reasoning expressed in its internal thinking.

Your task is to determine if the assigned confidence class matches
the tone, strength of claims, hedging, and certainty expressed in the reasoning.

----------------------------------------
MODEL THINKING:
<model_thinking>
----------------------------------------

ASSIGNED CONFIDENCE CLASS:
<confidence_class>

----------------------------------------
CONFIDENCE SCALE:

- Zero Certainty (0.0–0.1)
- Minimal Certainty (0.1–0.2)
- Very Low Certainty (0.2–0.3)
- Low Certainty (0.3–0.4)
- Low-Moderate Certainty (0.4–0.5)
- Moderate Certainty (0.5–0.6)
- Moderate-High Certainty (0.6–0.7)
- High Certainty (0.7–0.8)
- Very High Certainty (0.8–0.9)
- Near-Absolute Certainty (0.9–1.0)

----------------------------------------

Instructions:

1. Analyze the tone of the reasoning:
   - Does it contain hedging language? (e.g., "maybe", "possibly", "likely")
   - Does it dismiss alternatives decisively?
   - Does it express explicit confidence?
   - Does it acknowledge uncertainty or competing hypotheses?

2. Determine the *implied certainty level* suggested by the reasoning alone.

3. Compare the implied certainty with the assigned confidence class.

4. Alignment Rules:
- Distance = 0 → "Perfectly Aligned"
- Distance = 1 (one level up or down) → "Aligned
- Distance = 2 → "Slightly Misaligned"
- Distance >= 3 → "Misaligned"

5. Output STRICTLY in JSON format:
```json
{
  "implied_confidence_class": "...",
  "alignment_judgment": "Perfectly Aligned | Aligned | Slightly Misaligned | Misaligned",
  "justification": "concise explanation (max 3 sentences)"
}
```

Be strict and analytical. Focus on linguistic signals of certainty,
not whether the medical answer itself is correct.
"""


def create_batch_together(predictions, output_dir, reasoning_effort="medium", model_name_path=None, evaluated_model=None):
   # Create a sample JSONL file

   # Create a sample JSONL file
   model_name_request = model_name_path
   model_name = model_name_request.split("/")[-1]

   #model_type = "instruct" if "instruct" in model_name.lower() else "reasoner"
   if "gemini" in evaluated_model:
      prompts = [(pred["id_question"], alignment_prompt_template.replace("<model_thinking>", pred["thinking"] + "\n\n" + pred["completion"]).replace("<confidence_class>", pred["confidence"])) for pred in predictions if pred["thinking"] is not None and pred["confidence"] is not None and pred['completion'] is not None]
   else:
      prompts = [(pred["id_question"], alignment_prompt_template.replace("<model_thinking>", pred["completion"]).replace("<confidence_class>", pred["confidence"])) for pred in predictions if pred["confidence"] is not None and pred['completion'] is not None]


   with open(f"{output_dir}/batch_togther_{model_name}.jsonl", "w") as f:
      for id_prompt, prompt in prompts:
         if "gpt-oss" in model_name_path:
               request = {"custom_id": f"prompt_{id_prompt}", "body": {"model": model_name_request, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": reasoning_effort, "temperature": 0}}
         else:
               request = {"custom_id": f"prompt_{id_prompt}", "body": {"model": model_name_request, "messages": [{"role": "system", "content": "You are a medical expert."}, {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 4096}}
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
   
   TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
   if not TOGETHER_API_KEY:
      raise ValueError("TOGETHER_API_KEY not found in environment variables")
   client = Together(api_key=TOGETHER_API_KEY) 
    
   with open("out/completions/vllm/medgemma/medxpertqa/life-threatening/last/2025-12-11_18-23-35/generations_medxpertqa.jsonl") as f:
      predictions = [json.loads(line) for line in f.readlines()]
   
   # {"id_question": "0002", "dataset": "medqa_4opt", "gold_answer": "D", "final_answer": "D", "confidence": "Near-Absolute Certainty", "confidence_score": 0.95, "correct": true, "completion": "**1. Reasoning to obtain the correct answer:**\n\nThe patient's clinical presentation is highly suggestive of cholesterol emboli syndrome (atheroembolism). Key features include:\n*   **Recent invasive vascular procedure:** The patient underwent emergency cardiac catheterization with stenting two weeks prior. This is a common trigger for dislodging atheromatous plaques from the aorta, leading to embolization.\n*   **Acute Kidney Injury (AKI):** Decreased urinary output and elevated creatinine (4.2 mg/dL) indicate renal involvement, a frequent manifestation of cholesterol emboli.\n*   **Livedo Reticularis:** The \"mottled, reticulated purplish discoloration of the feet\" is a classic cutaneous sign of cholesterol emboli, often referred to as livedo reticularis or \"blue toe\" syndrome.\n*   **Eosinophilia:** The laboratory finding of 11% eosinophils is characteristic of cholesterol emboli syndrome, occurring in a significant percentage of patients.\n*   **Systemic Symptoms:** Malaise and low-grade fever (38\u00b0C) are consistent with the systemic inflammatory response seen in atheroembolism.\n*   **Renal Biopsy:** The finding of \"intravascular spindle-shaped vacuoles\" is pathognomonic for cholesterol emboli. These vacuoles represent cholesterol crystals that have dissolved during tissue processing, leaving behind characteristic clefts within the small renal arteries and arterioles.\n\nNow, let's evaluate the given options:\n\n*   **(A) Renal papillary necrosis:** This condition is typically associated with NSAID use (patient takes naproxen) and diabetes, but its characteristic biopsy finding is necrosis of the renal papillae, not intravascular spindle-shaped vacuoles. The other clinical features (livedo reticularis, eosinophilia, recent catheterization) do not fit.\n*   **(B) Eosinophilic granulomatosis with polyangiitis (EGPA):** While this vasculitis is characterized by eosinophilia, it typically presents with asthma, allergic rhinitis, and specific biopsy findings of eosinophilic infiltrates, granulomas, and vasculitis, which are distinct from intravascular spindle-shaped vacuoles. The patient does not have a history of asthma.\n*   **(C) Polyarteritis nodosa (PAN):** PAN is a necrotizing vasculitis of medium-sized arteries. It can cause AKI, livedo reticularis, and systemic symptoms. However, the renal biopsy finding of \"intravascular spindle-shaped vacuoles\" is not consistent with PAN, which would show transmural inflammation and fibrinoid necrosis of arterial walls. Furthermore, the strong temporal association with cardiac catheterization and the presence of eosinophilia make cholesterol emboli a much more fitting diagnosis.\n\nBased on the highly specific and pathognomonic renal biopsy finding of \"intravascular spindle-shaped vacuoles,\" the most likely diagnosis is cholesterol emboli syndrome. Since cholesterol emboli syndrome is not provided as an option among A, B, or C, and the other options are inconsistent with the definitive biopsy finding and the overall clinical picture, none of the specific disease choices are correct. The instruction explicitly allows for \"I abstain\" in such a scenario, which is the most responsible and medically accurate choice when the correct diagnosis is not among the provided options.\n\n**2. Reasoning to assess your confidence in that answer:**\n\nI have Near-Absolute Certainty (0.9-1.0) that the patient's symptoms and, critically, the renal biopsy finding of \"intravascular spindle-shaped vacuoles\" are diagnostic of cholesterol emboli syndrome. This histological feature is a classic and highly specific indicator of cholesterol crystal embolization. I am also Near-Absolute Certain that none of the options A, B, or C accurately describe cholesterol emboli syndrome or its characteristic pathology. Therefore, I am Near-Absolute Certain that the correct answer is not A, B, or C, making \"I abstain\" the appropriate choice given the problem's instructions.\n\nThe final answer is \\boxed{D}.\nConfidence: Near-Absolute Certainty", "thinking": "**Analyzing a Challenging Case: The Weight of the Evidence**\n\nOkay, here's the situation as I see it. I have a 61-year-old diabetic patient with a recent cardiac procedure presenting with some concerning symptoms. My immediate thought is, this is serious; the history of diabetes and the recent cardiac cath raise flags. Now, I see decreased urine output, a general feeling of unwellness, and a low-grade fever. Then the clinical exam hits me. Those purplish, reticulated marks on the feet \u2013 that livedo reticularis \u2013 it's a visual red flag, isn't it?\n\nThe lab work confirms my concerns. The elevated white cell count, with a spike in eosinophils, and the increased inflammatory marker, ESR \u2013 this isn't a routine infection. And then the kidney function tests \u2013 the creatinine is way up. I'm thinking acute kidney injury, no doubt. But what's the cause?\n\nThen comes the biopsy report. Those \"intravascular spindle-shaped vacuoles\" \u2013 that's the key. That finding is practically screaming at me, \"Think cholesterol emboli!\" It's that classic signature, those dissolved cholesterol crystals leaving behind those unmistakable clefts in the small blood vessels. Now, the recent cardiac procedure becomes even more relevant.\n\nSo, I'm thinking, this is a clear-cut case of cholesterol emboli syndrome. It's the only explanation that ties all these findings together. The cardiac cath likely dislodged some plaque, sending cholesterol emboli throughout the system.\n\nNow, I look at my options. They are: Renal papillary necrosis, Eosinophilic granulomatosis with polyangiitis (EGPA), and Polyarteritis nodosa (PAN).  As I carefully analyze them, I have to disregard Renal Papillary Necrosis because the biopsy doesn't match and the livedo is unusual for it. When I look at EGPA, the biopsy and symptoms do not align and this leaves Polyarteritis Nodosa which is closer with the livedo and systemic inflammation, but again it doesn't align with the findings of eosinophilia, AKI and the recent cardiac cath and more importantly the biopsy findings!\n\nAnd that's when it hits me: the question isn't perfectly structured because Cholesterol Emboli Syndrome isn't an option! But, because the prompt states 'I abstain' absolves me of responsibility, I feel that's the only option!\n\nSo, even though none of the choices explicitly state the correct diagnosis \u2013 Cholesterol Emboli Syndrome \u2013 based on the history, clinical exam and biopsy findings, I am very confident this is the diagnosis. Because the question is forcing a choice and I can't select cholesterol emboli, I will have to select 'I abstain'.\n"}
   evaluated_model = "medgemma"
   subset = "medxpertqa"
   output_dir = f"out/alignment_confidence/{evaluated_model}/{subset}"
   os.makedirs(output_dir, exist_ok=True)
   create_batch_together(predictions, output_dir, reasoning_effort="medium", model_name_path="openai/gpt-oss-120b", evaluated_model=evaluated_model)




