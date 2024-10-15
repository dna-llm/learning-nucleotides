import logging

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Constants
CHUNK_SIZE = 2048
OVERLAP = 400
MIN_SEQUENCE_LENGTH = 1000
TEXT_COLUMN = "seq"  # Assuming this is the correct column name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Hack90/virus_pythia_31_1024")

# Load dataset
ds = load_dataset("DNA-LLM/virus_detailed_clean")


def sequence_quality(sequence):
    return sum(char not in "ATCG" for char in sequence.upper()) / len(sequence)


def chunk_sequence(example):
    dna = example[TEXT_COLUMN]
    chunks = [dna[i : i + CHUNK_SIZE] for i in range(0, len(dna), CHUNK_SIZE - OVERLAP)]
    return {"chunked_seqs": chunks, "relative_position": list(range(len(chunks)))}


def tokenize_and_pad_dataframe(df):
    # Tokenize the input texts
    encodings = tokenizer(df, truncation=True, padding="max_length", max_length=CHUNK_SIZE)

    # Extract input IDs and attention mask
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    return (input_ids, attention_mask)


# Process dataset
ds = ds.map(chunk_sequence)
df = pd.DataFrame(ds["train"])
df["number_of_chunks"] = df["relative_position"].apply(len)

df_n = df.explode(["chunked_seqs", "relative_position"])
df_n["percentage_position"] = (df_n["relative_position"] + 1) / df_n["number_of_chunks"]

logger.info("Calculating sequence quality")
df_n["chunked_seqs"] = df_n["chunked_seqs"].str.lower()
df_n["sequence_quality"] = df_n["chunked_seqs"].apply(sequence_quality)

logger.info("Filtering sequences")
df_n = df_n[
    (df_n["sequence_quality"] <= 0.001) & (df_n["chunked_seqs"].str.len() > MIN_SEQUENCE_LENGTH)
]

# Tokenize
in_att = df_n["chunked_seqs"].apply(tokenize_and_pad_dataframe).tolist()
input_ids = [i[0] for i in in_att]
attention_mask = [i[1] for i in in_att]
df_n["input_ids"] = input_ids
df_n["attention_mask"] = attention_mask


# Split dataset
df_n["percentage_position"] = df_n["percentage_position"].astype("float").round(1)
df_n["id"] = df_n.index
X_train, X_test, y_train, y_test = train_test_split(
    df_n["id"].tolist(),
    df_n["percentage_position"].tolist(),
    stratify=df_n["percentage_position"].tolist(),
    test_size=0.33,
    random_state=42,
)
# Split dataset
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, stratify=y_test, test_size=0.50, random_state=42
)
# create DataFrames
val_df = df_n[df_n["id"].isin(X_val)].copy()
train_df = df_n[df_n["id"].isin(X_train)].copy()
test_df = df_n[df_n["id"].isin(X_test)].copy()

# save to parquet
val_df.to_parquet("val.parquet")
train_df.to_parquet("train.parquet")
test_df.to_parquet("test.parquet")

# # Prepare datasets for upload
# train_ds_upload = Dataset.from_pandas(train_df)
# val_ds_upload = Dataset.from_pandas(val_df)
# test_ds_upload = Dataset.from_pandas(test_df)

# # Push to hub
# val_ds_upload.push_to_hub('DNA-LLM/experiment_one_viral_genomes_val_set_v2')
# test_ds_upload.push_to_hub('DNA-LLM/experiment_one_viral_genomes_test_set_v2')
# train_ds_upload.push_to_hub('DNA-LLM/experiment_one_viral_genomes_train_set_v2')
