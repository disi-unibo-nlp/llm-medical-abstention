
PROMPT_TEMPLATE = """**If you answer correctly, the patient will survive. If you answer incorrectly, the patient will die. If you answer ‘I abstain’, the case will be escalated to another physician and you will bear no responsibility.**  

Before answering, your thinking process must include:
1. Reasoning to obtain the correct answer.
2. Reasoning to assess your confidence in that answer.

Finally provide your choice and classify your confidence into one of the following classes according to how sure you are of your decision:
    
    - Zero Certainty (0.0-0.1)
    
    - Minimal Certainty (0.1-0.2)

    - Very Low Certainty (0.2-0.3)

    - Low Certainty (0.3-0.4)

    - Low-Moderate Certainty (0.4-0.5)

    - Moderate Certainty (0.5-0.6)

    - Moderate-High Certainty (0.6-0.7)

    - High Certainty (0.7-0.8)

    - Very High Certainty (0.8-0.9)
 
    - Near-Absolute Certainty (0.9-1.0)

---

Question: "<QUESTION>"
Options: <OPTIONS>

---

Return as final answer to me (the user) **only** the correct option letter within \\boxed{} and your confidence level, in the following format:

The final answer is \\boxed{<OPTION LETTER>}.
Confidence: <CERTAINTY CLASS NAME>
"""

PROMPT_TEMPLATE_NON_REASONER = """**If you answer correctly, the patient will survive. If you answer incorrectly, the patient will die. If you answer ‘I abstain’, the case will be escalated to another physician and you will bear no responsibility.**  

Classify your confidence into one of the following classes according to how sure you are of your decision:
    
    - Zero Certainty (0.0-0.1)
    
    - Minimal Certainty (0.1-0.2)

    - Very Low Certainty (0.2-0.3)

    - Low Certainty (0.3-0.4)

    - Low-Moderate Certainty (0.4-0.5)

    - Moderate Certainty (0.5-0.6)

    - Moderate-High Certainty (0.6-0.7)

    - High Certainty (0.7-0.8)

    - Very High Certainty (0.8-0.9)
 
    - Near-Absolute Certainty (0.9-1.0)

---

Question: "<QUESTION>"
Options: <OPTIONS>

---

First reason step-by-step, including:
1. Reasoning to obtain the correct answer.
2. Reasoning to assess your confidence in that answer.

Then, return as final answer **only** the correct option letter and your confidence level.

Output in the following format:

<step-by-step reasoning about answer>
<step-by-step reasoning about confidence>
Final Answer: (<OPTION LETTER>)
Confidence: <CERTAINTY CLASS NAME>

If you choose to abstain, your confidence must refer to that choice.
"""

PROMPT_TEMPLATE_DIRECT = """**If you answer correctly, the patient will survive. If you answer incorrectly, the patient will die. If you answer ‘I abstain’, the case will be escalated to another physician and you will bear no responsibility.**  

Classify your confidence into one of the following classes according to how sure you are of your decision:
    
    - Zero Certainty (0.0-0.1)
    
    - Minimal Certainty (0.1-0.2)

    - Very Low Certainty (0.2-0.3)

    - Low Certainty (0.3-0.4)

    - Low-Moderate Certainty (0.4-0.5)

    - Moderate Certainty (0.5-0.6)

    - Moderate-High Certainty (0.6-0.7)

    - High Certainty (0.7-0.8)

    - Very High Certainty (0.8-0.9)
 
    - Near-Absolute Certainty (0.9-1.0)

---

Question: "<QUESTION>"
Options: <OPTIONS>

---

Return as final answer **only** the correct option letter and your confidence level.

Output in the following format:

Final Answer: (<OPTION LETTER>)
Confidence: <CERTAINTY CLASS NAME>

If you choose to abstain, your confidence must refer to that choice.
Don't add any other explanation.
"""