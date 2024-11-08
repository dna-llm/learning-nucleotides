# %%
import json
import random

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.manifold import TSNE
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

import matplotlib.colors as colors


random.seed(42)
set_seed(42)

# %%
def generate_model_seqs(
    start_token,
    model,
    tokenizer,
    max_length=100,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    num_return_sequences=3,
):
    input_ids = tokenizer.encode(start_token, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    sample_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_length - input_ids.shape[1],  # Adjust for the length of the start token
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    sequences = [tokenizer.decode(output, skip_special_tokens=True) for output in sample_outputs]

    return sequences


def generate_tSNE_embedding(natural_sequences, generated_sequences, ax=None):
    all_sequences = natural_sequences + generated_sequences

    num_sequences = len(all_sequences)

    n_natural = len(natural_sequences)
    n_generated = len(generated_sequences)

    distance_matrix = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(num_sequences):
            if i != j:
                distance_matrix[i][j] = 1 - Levenshtein.ratio(all_sequences[i], all_sequences[j])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(distance_matrix)

    labels = ["Natural"] * n_natural + ["Generated"] * n_generated
    tsne_data = {"t-SNE 1": tsne_results[:, 0], "t-SNE 2": tsne_results[:, 1], "Label": labels}

    sns.jointplot(
        x="t-SNE 1",
        y="t-SNE 2",
        hue="Label",
        palette=["indianred", "steelblue"],
        hue_order=["Generated", "Natural"],
        data=tsne_data,
        ax=ax,
    )


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the percent GC content of a genome sequence.

    Parameters:
    sequence (str): The genome sequence.

    Returns:
    float: The percent GC content.
    """
    g_count = sequence.upper().count("G")
    c_count = sequence.upper().count("C")
    gc_count = g_count + c_count
    total_bases = len(sequence)

    if total_bases == 0:
        return 0.0

    percent_gc = (gc_count / total_bases) * 100
    return percent_gc


def calculate_gc_content_list(sequences: list) -> list:
    """
    Calculate the percent GC content for a list of genome sequences.

    Parameters:
    sequences (list): A list of genome sequences.

    Returns:
    list: A list of percent GC content for each sequence.
    """
    gc_contents = [calculate_gc_content(seq) for seq in sequences]
    return gc_contents


def plot_gc_content_boxplot(natural_gc_contents: list, generated_gc_contents: list):
    """
    Plot a box-and-whiskers plot of GC content percentages for natural and LM-generated sequences.

    Parameters:
    natural_gc_contents (list): A list of percent GC content for natural sequences.
    generated_gc_contents (list): A list of percent GC content for LM-generated sequences.
    """

    sns.set(style="whitegrid")
    data = {
        "GC Content": natural_gc_contents + generated_gc_contents,
        "Type": ["Natural"] * len(natural_gc_contents) + ["Generated"] * len(generated_gc_contents),
    }
    df = pd.DataFrame(data)

    sns.boxplot(
        x="Type",
        y="GC Content",
        data=df,
        palette={"Natural": "indianred", "Generated": "steelblue"},
        linewidth=2.5,
        width=0.6,
    )
    sns.stripplot(x="Type", y="GC Content", data=df, color="black", jitter=0.2, size=5, alpha=0.7)


def sample_sequences_from_test_set(num_samples=100):
    num_samples = min(num_samples, len(test_ds["train"]))

    sampled_indices = random.sample(range(len(test_ds["train"])), num_samples)
    sampled_sequences = [test_ds["train"][i] for i in sampled_indices]

    return sampled_sequences


def preprocess_sequences(sampled_sequences):
    """
    Split each sampled sequence into halves. First half is for conditional generation,
    Second half is to evaluate against LM-generated seq, lengths of each half split
    are to generate equal length sequences for fair comparison of GC content and
    sequence identity.

    """
    first_halves = []
    second_halves = []
    first_halves_lengths = []
    second_halves_lengths = []

    for i, sample in enumerate(sampled_sequences):
        seq = sample["sequence"]
        n = len(seq)

        first_half, second_half = seq[: n // 2], seq[n // 2 :]
        first_half_len, second_half_len = len(first_half), len(second_half)

        print("i:", i, "len first half:", first_half_len, "len second half:", second_half_len)

        first_halves.append(first_half)
        second_halves.append(second_half)

        first_halves_lengths.append(first_half_len)
        second_halves_lengths.append(second_half_len)

    return first_halves, second_halves, first_halves_lengths, second_halves_lengths


def generate_seq_from_halves(model, first_halves, second_halves_lengths):
    # Uncomment for denseformer
    # model.to('cuda', dtype=torch.bfloat16)
    generated_seqs = []

    tokenizer = AutoTokenizer.from_pretrained("Hack90/virus_pythia_14_1024_compliment")

    for i, half in enumerate(first_halves):
        if i % 10 == 0:
            print("sequence:", i + 1)
        input_ids = tokenizer.encode(half.lower(), return_tensors="pt")

        # Ensure input_ids are on the GPU --> uncomment for densefomer model
        # input_ids = input_ids.to('cuda', dtype=torch.long)

        outputs = model.generate(
            input_ids,
            max_new_tokens=second_halves_lengths[i],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[0], [1], [2]],
        )

        output = tokenizer.decode(outputs.tolist()[0])
        generated_seqs.append(output)

    return generated_seqs


def generate_tSNE_embedding_dynamic(
    natural_sequences, generated_sequences_list, model_names, ax=None
):
    """
    Generate and plot t-SNE embedding for natural sequences and generated sequences from multiple models.

    Parameters:
    natural_sequences (list): A list of natural sequences.
    generated_sequences_list (list of lists): A list containing lists of generated sequences from different models.
    model_names (list): A list of names of the models that generated the sequences.
    ax (matplotlib.axes.Axes, optional): An axis object to plot on. Defaults to None.
    """

    sns.reset_orig()

    generated_sequences = [seq for sublist in generated_sequences_list for seq in sublist]
    all_sequences = natural_sequences + generated_sequences

    num_sequences = len(all_sequences)
    n_natural = len(natural_sequences)
    n_generated_per_model = [len(sublist) for sublist in generated_sequences_list]

    # Compute the distance matrix
    distance_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(num_sequences):
            if i != j:
                distance_matrix[i][j] = 1 - Levenshtein.ratio(all_sequences[i], all_sequences[j])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(distance_matrix)

    labels = ["Natural"] * n_natural
    for i, model_name in enumerate(model_names):
        labels.extend([model_name] * n_generated_per_model[i])

    tsne_data = {"t-SNE 1": tsne_results[:, 0], "t-SNE 2": tsne_results[:, 1], "Label": labels}

    # Convert to DataFrame
    tsne_df = pd.DataFrame(tsne_data)

    sns.jointplot(x="t-SNE 1", y="t-SNE 2", hue="Label", data=tsne_df, palette=palette)


def plot_gc_content_boxplot_dynamic(
    natural_gc_contents: list, generated_gc_contents_list: list, model_names: list
):
    """
    Plot side-by-side boxplots of GC content percentages for natural sequences and sequences generated by multiple models.

    Parameters:
    natural_gc_contents (list): A list of percent GC content for natural sequences.
    generated_gc_contents_list (list of lists): A list containing lists of percent GC content for sequences generated by different models.
    model_names (list): A list of names of the models that generated the sequences.
    """
    sns.set(style="whitegrid")

    gc_contents = natural_gc_contents
    types = ["Natural"] * len(natural_gc_contents)

    for i, model_gc_contents in enumerate(generated_gc_contents_list):
        gc_contents.extend(model_gc_contents)
        types.extend([model_names[i]] * len(model_gc_contents))

    data = {"GC Content": gc_contents, "Type": types}
    df = pd.DataFrame(data)

    palette = sns.color_palette("tab10", len(model_names) + 1)
    sns.boxplot(x="Type", y="GC Content", data=df, palette=palette, linewidth=2.5, width=0.6)
    sns.stripplot(x="Type", y="GC Content", data=df, color="black", jitter=0.2, size=5, alpha=0.7)

    plt.title("GC Content Box-and-Whiskers Plot")
    plt.ylabel("Percent GC Content")
    plt.xlabel("Sequence Type")
    plt.xticks(rotation=45)


# !huggingface-cli login
# %%
test_ds = load_dataset("Hack90/experiment_one_viral_genomes_test_set")
sampled_sequences = sample_sequences_from_test_set()
first_halves, second_halves, first_halves_lengths, second_halves_lengths = preprocess_sequences(
    sampled_sequences
)

api = HfApi()

MODEL_NAME = "togethercomputer/evo-1-8k-base"

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, revision="1.1_fix")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, revision="1.1_fix")

model_types = ["pythia", "denseformer", "evo"]
param_sizes = ["14", "31", "70", "160", "410"]
losses = [
    "cross_entropy",
    "complement",
    "headless",
    "2d_representation_GaussianPlusCE",
    "2d_representation_MSEPlusCE",
    "two_d",
]

model_name_to_generated_seqs = {}

all_models = []

models = api.list_models(author="DNA-LLM")
for m in models:
    all_models.append(m.id)

for model_type in model_types:
    for param_size in param_sizes:
        for loss in losses:
            model_string = f"DNA-LLM/virus_{model_type}_{param_size}_1024_{loss}"

            if model_string in all_models:
                # Pythia 31 1024 Headless doesn't have config.json
                if model_string == "DNA-LLM/virus_pythia_31_1024_headless":
                    model = AutoModelForCausalLM.from_pretrained("/content/checkpoint-7000")
                elif model_string == "DNA-LLM/virus_pythia_31_1024_compliment":
                    model = AutoModelForCausalLM.from_pretrained("/content/checkpoint-15000")
                else:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_string, trust_remote_code=True
                        )
                    except Exception:
                        print(f"Failed to download {model_string}")
                        continue

                model_name_short = f"{model_type}_{param_size}_{loss}"
                model_name_to_generated_seqs[model_name_short] = generate_seq_from_halves(
                    model, first_halves, second_halves_lengths
                )

# Convert and write JSON object to file
# with open("results.json", "w") as outfile:
#    json.dump(model_name_to_generated_seqs, outfile)

# with open("results.json") as file:
#  model_name_to_generated_seqs = json.load(file)

with open("outputs/pythia_results.json", "r") as file:
    p = json.load(file)

with open("outputs/evo_31_1024_cross_entropy_results.json", "r") as file:
    e = json.load(file)

# with open("outputs/denseformer_31_1024_cross_entropy_results.json", "r") as file:
#     d = json.load(file)
# %%
natural_seqs = [half.lower() for half in second_halves]

plt.figure(figsize=(12, 12))

p_31_losses = [
    p["pythia_31_cross_entropy"],
    p["pythia_31_compliment"],
    p["pythia_31_headless"],
    p["pythia_31_2d_representation_GaussianPlusCE"],
    # p["pythia_31_2d_representation_MSEPlusCE"],
]

generate_tSNE_embedding_dynamic(
    natural_seqs,
    p_31_losses,
    ["Pythia CE", "Pythia Compl.", "Pythia Headless", "Pythia 2D"] #, "Pythia MSE+CE"],
)

plt.savefig("outputs/pythia_loss_tsne.png", format="png")

plt.figure(figsize=(12, 12))

natural_gc_contents = calculate_gc_content_list(natural_seqs)
generated_gc_contents_model_1 = calculate_gc_content_list(p["pythia_31_cross_entropy"])
generated_gc_contents_model_2 = calculate_gc_content_list(p["pythia_31_compliment"])
generated_gc_contents_model_3 = calculate_gc_content_list(p["pythia_31_headless"])
generated_gc_contents_model_4 = calculate_gc_content_list(
    p["pythia_31_2d_representation_GaussianPlusCE"]
)
# generated_gc_contents_model_5 = calculate_gc_content_list(
#     p["pythia_31_2d_representation_MSEPlusCE"]
# )

generated_gc_contents_list = [
    generated_gc_contents_model_1,
    generated_gc_contents_model_2,
    generated_gc_contents_model_3,
    generated_gc_contents_model_4,
    # generated_gc_contents_model_5,
]

model_names = [
    "Pythia 31M CE",
    "Pythia 31M Complement",
    "Pythia 31M Headless",
    "Pythia 31M 2D"   #2DGaussian+CE",
    # "Pythia 31M MSE+CE",
]

plot_gc_content_boxplot_dynamic(natural_gc_contents, generated_gc_contents_list, model_names)

plt.savefig("outputs/pythia_loss_gc.png", format="png")

# %%
natural_seqs = [half.lower() for half in second_halves]

plt.figure(figsize=(12, 12))

p_param_losses = [
    p["pythia_14_compliment"],
    p["pythia_31_compliment"],
    p["pythia_70_compliment"],
    p["pythia_160_compliment"],
    p["pythia_410_compliment"],
]

generate_tSNE_embedding_dynamic(
    natural_seqs,
    p_param_losses,
    ["Pythia 14M", "Pythia 31M", "Pythia 70M", "Pythia 160M", "Pythia 410M"],
)

plt.savefig("outputs/pythia_param_tsne_14M.png", format="png")

plt.figure(figsize=(12, 12))

natural_gc_contents = calculate_gc_content_list(natural_seqs)
generated_gc_contents_model_0 = calculate_gc_content_list(p["pythia_14_compliment"])
generated_gc_contents_model_1 = calculate_gc_content_list(p["pythia_31_compliment"])
generated_gc_contents_model_2 = calculate_gc_content_list(p["pythia_70_compliment"])
generated_gc_contents_model_3 = calculate_gc_content_list(p["pythia_160_compliment"])
generated_gc_contents_model_4 = calculate_gc_content_list(p["pythia_410_compliment"])


generated_gc_contents_list = [
    generated_gc_contents_model_0,
    generated_gc_contents_model_1,
    generated_gc_contents_model_2,
    generated_gc_contents_model_3,
    generated_gc_contents_model_4,
]

model_names = ["Pythia 14M", "Pythia 31M", "Pythia 70M", "Pythia 160M", "Pythia 410M"]

plot_gc_content_boxplot_dynamic(natural_gc_contents, generated_gc_contents_list, model_names)

plt.savefig("outputs/pythia_param_gc_14M.png", format="png")

natural_seqs = [half.lower() for half in second_halves]

plt.figure(figsize=(12, 12))

model_type_seqs = [
    p["pythia_31_cross_entropy"],
    e["evo_31_cross_entropy"],
    # d["denseformer_31_cross_entropy"],
]

generate_tSNE_embedding_dynamic(natural_seqs, model_type_seqs, ["Pythia", "Evo", "Denseformer"])

plt.savefig("outputs/pythia_model_type_tsne.png", format="png")

plt.figure(figsize=(12, 12))

natural_gc_contents = calculate_gc_content_list(natural_seqs)
generated_gc_contents_model_1 = calculate_gc_content_list(p["pythia_31_cross_entropy"])
generated_gc_contents_model_2 = calculate_gc_content_list(e["evo_31_cross_entropy"])
# generated_gc_contents_model_3 = calculate_gc_content_list(d["denseformer_31_cross_entropy"])

generated_gc_contents_list = [
    generated_gc_contents_model_1,
    generated_gc_contents_model_2,
    generated_gc_contents_model_3,
]

model_names = ["Pythia", "Evo", "Denseformer"]

plot_gc_content_boxplot_dynamic(natural_gc_contents, generated_gc_contents_list, model_names)

plt.savefig("outputs/pythia_model_type_gc.png", format="png")
