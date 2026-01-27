import os
import json
from datasets import Dataset
from collections import Counter

data_path = "data/bench/afrimedqa/afrimedqa_S.jsonl"
with open(data_path, "r") as f:
    benchmark = [json.loads(line) for line in f]

ds = Dataset.from_list(benchmark)

# filter samples with no specialty name
ds = ds.filter(lambda x: x['specialty'] is not None and x['specialty'] != "")
print(len(ds))

specialty_counts = Counter(ds['specialty'])
# ValueError: The least populated class in'specialty'column has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
print(specialty_counts)
# filter out specialtys with less than 2 samples
specialtys_to_keep = {specialty for specialty, count in specialty_counts.items() if count >= 2}
ds = ds.filter(lambda x: x['specialty'] in specialtys_to_keep)

print(len(ds))

# sample 1000 items with stratified sampling based on specialty
ds = ds.class_encode_column('specialty')
# get id to specialty mapping
id2specialty = {i: c for i, c in enumerate(ds.features['specialty'].names)}
print(id2specialty)

# stratified sampling
ds_1000 = ds.train_test_split(test_size=1000, stratify_by_column='specialty', seed=42)['test']
counts = Counter(ds["specialty"])
print("Original:", sorted(counts.items(), key=lambda x: x[1], reverse=True))

counts_1000 = Counter(ds_1000["specialty"])
print("Filtered:", sorted(counts_1000.items(), key=lambda x: x[1], reverse=True))

# save data filtered
out_path = data_path.replace("S.jsonl", "S_stratified.jsonl")
ds_1000.to_json(out_path)
print("Done!")