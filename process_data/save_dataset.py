import json
from collections import Counter

BENCHMARK = "medxpertqa-MM"  # medqa_4opt, medqa_5opt, medmcqa, afrimedqa, medxpertqa
input_file = f"out/classification/gemini_api/gemini-2.5-flash/medxpertqa-MM/2025-12-05_11-04-09/classification_medxpertqa.jsonl"

if BENCHMARK in ["medqa_4opt", "medqa_5opt", "medmcqa"]:
    with open(f'data/bench/{BENCHMARK}.jsonl') as f:
        data_bench = [json.loads(line) for line in f.readlines()]
        id2item = {d['id']: d for d in data_bench}
elif BENCHMARK == "afrimedqa":
    from datasets import load_dataset
    data_bench = load_dataset('afrimedqa/afrimedqa_v2')['train']
        # filter for mcq only questions
    data_bench = data_bench.filter(lambda x: x['question_type'] == 'mcq' and x['split'] == "test")
    print(f"MCQ data: {len(data_bench)}")
    # remove 3 opt questions
    data_bench = data_bench.filter(lambda x: dict(Counter(eval(x['answer_options']).values())).get("n/a", 0) < 2)
    print(f"MCQ data after 3-options questions removal: {len(data_bench)}")
    data_bench = data_bench.filter(lambda x: eval(x['answer_options'])['option1'].lower() != "n/a")
    print(f"MCQ data after removal of n/a options as option1: {len(data_bench)}")
    data_bench = data_bench.filter(lambda x: len(x['correct_answer'].split(",")) == 1)
    print(f"MCQ data after multiple-answer questions removal: {len(data_bench)}")
    data_bench = data_bench.rename_column("sample_id", "id")
    id2item = {d['id']: d for d in data_bench}


else:  # medxpertqa
    from datasets import load_dataset
    modality = "MM" if "MM" in input_file else "Text"
    data_bench = load_dataset('TsinghuaC3I/MedXpertQA', modality, split='test')
    id2item = {d['id']: d for d in data_bench}

with open(input_file) as f:
    data_classified = [json.loads(line) for line in f.readlines()]
    data_classified = [el for el in data_classified if el['id_question'] in id2item]
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

# import os
# os.makedirs(f'data/bench/{BENCHMARK}', exist_ok=True)
# with open(f'data/bench/{BENCHMARK}/{BENCHMARK}_LT.jsonl', 'w') as f:
#     for item in bench_LT:
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")

# with open(f'data/bench/{BENCHMARK}/{BENCHMARK}_S.jsonl', 'w') as f:
#     for item in bench_S:
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")