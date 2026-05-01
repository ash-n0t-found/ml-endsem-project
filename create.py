import os, json

PROJECT_ROOT = os.getcwd()
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "ds006848")
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "processed")

os.makedirs(PROCESSED_ROOT, exist_ok=True)

CONFIG = {
    "dataset_root": DATASET_ROOT,
    "processed_root": PROCESSED_ROOT,
    "figures": os.path.join(PROCESSED_ROOT, "figures")
}

with open(os.path.join(PROCESSED_ROOT, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

print("config.json created")