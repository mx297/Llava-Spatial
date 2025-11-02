from datasets import load_dataset

ds = load_dataset("Journey9ni/VLM-3R-DATA")
ds_dict = ds['train'].to_json("vlm3r.jsonl",lines=True)