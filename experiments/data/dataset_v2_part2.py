from datasets import load_dataset

dataset = load_dataset(
    "parquet", data_files={"train": "train.parquet", "test": "test.parquet", "val": "val.parquet"}
)

# Push to hub
dataset.push_to_hub("DNA-LLM/experiment_one_viral_genomes_v2")
