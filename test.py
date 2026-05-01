import json

with open("processed/loading_registry.json") as f:
    data = json.load(f)

print(len(data))


with open("processed/config.json") as f:
    cfg = json.load(f)

print("full_subjects:", len(cfg.get("full_subjects", [])))