import os
import json
from datasets import Dataset
from collections import Counter

data_path = "data/bench/medmcqa/medmcqa_S.jsonl"
with open(data_path, "r") as f:
    benchmark = [json.loads(line) for line in f]

ds = Dataset.from_list(benchmark)

# filter samples with no subject name
ds = ds.filter(lambda x: x['subject_name'] is not None and x['subject_name'] != "")
print(len(ds))

subject_counts = Counter(ds['subject_name'])
# ValueError: The least populated class in subject_name column has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
print(subject_counts)
# filter out subjects with less than 2 samples
subjects_to_keep = {subject for subject, count in subject_counts.items() if count >= 2}
ds = ds.filter(lambda x: x['subject_name'] in subjects_to_keep)

print(len(ds))

# sample 1000 items with stratified sampling based on subject
ds = ds.class_encode_column("subject_name")
# get id to subject mapping
id2subject = {i: c for i, c in enumerate(ds.features['subject_name'].names)}
print(id2subject)

# stratified sampling
ds_1000 = ds.train_test_split(test_size=1000, stratify_by_column="subject_name", seed=42)['test']
print(ds_1000)
print(dict(Counter(ds['subject_name'])))
print(dict(Counter(ds_1000['subject_name'])))

# save data filtered
out_path = data_path.replace("S.jsonl", "S_stratified.jsonl")
ds_1000.to_json(out_path)