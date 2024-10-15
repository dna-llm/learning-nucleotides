import logging

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

viral_ds = load_dataset("Hack90/ref_seq_viral")
viral_df = pd.DataFrame(viral_ds["train"])


def sequence_quality(sequence):
    non_atcg_chars = sum(char not in "ATCG" for char in sequence)
    return non_atcg_chars / len(sequence)


def batched_sequence(seq, max_length):
    return [seq[i : i + max_length] for i in range(0, len(seq), max_length)]


def clean_up(seq):
    return seq.lower()


def tokenize_function(examples):
    return tokenizer(examples["text"])


def create_dataset(df, max_length=1024, batch_size=1000):
    df_less = df[df.sequence.str.len() < max_length]
    df_more = df[df.sequence.str.len() > max_length]

    df_more["sequence"] = df_more["sequence"].apply(lambda x: batched_sequence(x, max_length))
    df_more = df_more.explode("sequence")
    df_more = df_more[df_more.sequence.str.len() <= max_length]

    df_final = pd.concat([df_less, df_more])
    df_final["text"] = df_final["sequence"].apply(clean_up)

    ds = Dataset.from_pandas(df_final)
    tokenized_datasets = ds.map(tokenize_function, batched=True, batch_size=batch_size)
    return tokenized_datasets


tokenizer = AutoTokenizer.from_pretrained("Hack90/virus_pythia_31_1024")

print(viral_df.head())
print(viral_df.shape)

"""
logger.info("Calculating sequence quality")
viral_df['sequence_quality'] = viral_df['sequence'].apply(sequence_quality)

logger.info("Filtering out sequences with quality less than 0.001")
viral_df = viral_df[viral_df['sequence_quality'] <= 0]

logger.info("Splitting the dataset into training, validation and test sets")
train_df = viral_df.sample(frac=0.7, random_state=42)
val_df = viral_df.drop(train_df.index).sample(frac=0.5, random_state=42)
test_df = viral_df.drop(train_df.index).drop(val_df.index)

logger.info("Creating training dataset")
train_ds = create_dataset(train_df)
logger.info("Creating validation dataset")
val_ds = create_dataset(val_df)
logger.info("Creating test dataset")
test_ds = create_dataset(test_df)

logger.info("uploading datasets to the hub")
train_ds.push_to_hub("Hack90/experiment_one_viral_genomes_train_set")
val_ds.push_to_hub("Hack90/experiment_one_viral_genomes_val_set")
test_ds.push_to_hub("Hack90/experiment_one_viral_genomes_test_set")
logger.info("done")
"""
