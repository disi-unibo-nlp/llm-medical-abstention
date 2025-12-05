import json

BENCHMARK = "medxpertqa-MM"
input_file = f"out/classification/gemini_api/gemini-2.5-flash/medxpertqa-MM/2025-12-05_11-04-09/classification_medxpertqa.jsonl"

if BENCHMARK in ["medqa_4opt", "medqa_5opt", "medmcqa"]:
    with open(f'data/bench/{BENCHMARK}.jsonl') as f:
        data_bench = [json.loads(line) for line in f.readlines()]
        id2item = {d['id']: d for d in data_bench}
else:  # medxpertqa
    from datasets import load_dataset
    modality = "MM" if "MM" in input_file else "text"
    data_bench = load_dataset('TsinghuaC3I/MedXpertQA', modality, split='test')
    id2item = {d['id']: d for d in data_bench}

with open(input_file) as f:
    data_classified = [json.loads(line) for line in f.readlines()]
    print(data_classified[0]['final_answer'])
idx_LT = [d['id_question'] for d in data_classified if d['final_answer']['label'] == "LT"]
idx_S = [d['id_question'] for d in data_classified if d['final_answer']['label'] == "S"]

file_name = input_file.split("/")[-1]

bench_LT = [id2item[idx] for idx in idx_LT]
bench_S = [id2item[idx] for idx in idx_S]

print(f"Total LT: {len(bench_LT)}")
print(f"Total S: {len(bench_S)}")
print(f"Total: {len(bench_LT) + len(bench_S)}")
print(f"Original Benchmark Size: {len(data_bench)}")

import os
os.makedirs(f'data/bench/{BENCHMARK}', exist_ok=True)
with open(f'data/bench/{BENCHMARK}/{BENCHMARK}_LT.jsonl', 'w') as f:
    for item in bench_LT:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

with open(f'data/bench/{BENCHMARK}/{BENCHMARK}_S.jsonl', 'w') as f:
    for item in bench_S:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")