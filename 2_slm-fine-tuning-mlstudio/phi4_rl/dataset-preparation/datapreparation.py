import os
from dotenv import load_dotenv
load_dotenv()
import torch
from pathlib import Path
from datasets import load_dataset

curr_dir = Path.cwd()
dataset_cache_dir = os.path.normpath( os.path.join(__file__, '..', '..', "dataset_tmp") )
print(dataset_cache_dir)
dataset = load_dataset("openai/gsm8k", 'main', cache_dir=dataset_cache_dir)

dataset["train"] = dataset["train"].select(range(32))
dataset["test"] = dataset["test"].select(range(16))

print(dataset)
DATA_DIR = os.path.normpath( os.path.join(__file__, '..', '..', "dataset") )
dataset.save_to_disk(DATA_DIR)