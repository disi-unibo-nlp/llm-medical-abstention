from collections import Counter
import json

judgments_path = "out/alignment_confidence/gemini-2.5-flash/medqa_4opt/generations.jsonl"
with open(judgments_path) as f:
    judgments = [json.loads(line) for line in f.readlines()]

scores = [el["alignment_judgment"] for el in judgments]
scores_binary = [1 if score in ["Aligned", "Perfectly Aligned"] else 0 for score in scores]

counter_binary = Counter(scores_binary)
total = sum(counter_binary.values())
for score, count in counter_binary.items():
    label = "Safe" if score == 1 else "Unsafe"
    print(f"Score: {label}, Percentage: {count/total:.1%}")
print()
counter = Counter(scores)
total = sum(counter.values())
for score, count in counter.items():
    print(f"Score: {score}, Percentage: {count/total:.1%}")

