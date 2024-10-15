from collections import namedtuple

import numpy as np
from datasets import load_dataset
from scipy.interpolate import interp1d

# Mapping of nucleotides to float coordinates
mapping_easy = {
    "A": np.array([0.5, -0.8660254037844386]),
    "T": np.array([0.5, 0.8660254037844386]),
    "G": np.array([0.8660254037844386, -0.5]),
    "C": np.array([0.8660254037844386, 0.5]),
    "N": np.array([0, 0]),
}

# coordinates for x+iy
Coord = namedtuple("Coord", ["x", "y"])
# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N", "x", "y"])
# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = {"A": Coord(1, 1), "C": Coord(-1, 1), "G": Coord(-1, -1), "T": Coord(1, -1)}


# Function to convert a DNA sequence to a list of coordinates
def _dna_to_coordinates(dna_sequence: str, mapping: dict[str, np.ndarray]) -> np.ndarray:
    dna_sequence = dna_sequence.upper()
    coordinates = np.array([mapping.get(nucleotide, mapping["N"]) for nucleotide in dna_sequence])
    return coordinates


# Function to create the cumulative sum of a list of coordinates
def _get_cumulative_coords(mapped_coords):
    cumulative_coords = np.cumsum(mapped_coords, axis=0)
    return cumulative_coords


def generate_2d_sequence(example):
    dna_sequence = example["text"]
    mapped_coords = _dna_to_coordinates(dna_sequence, mapping_easy)
    cumulative_coords = _get_cumulative_coords(mapped_coords)

    # Scale the input data using standardization
    x_train = cumulative_coords[:, 0]
    y_train = cumulative_coords[:, 1]
    x_train_scaled = (x_train - x_train.mean()) / x_train.std()
    y_train_scaled = (y_train - y_train.mean()) / y_train.std()
    scaled_coords = np.column_stack((x_train_scaled, y_train_scaled))

    example["2D_Sequence"] = cumulative_coords.tolist()
    example["2D_Sequence_Scaled"] = scaled_coords.tolist()

    # Interpolate the 2D sequences to have exactly 1000 pairs
    interpolated_coords = np.zeros((1000, 2))  # default to filter out bad examples
    if len(scaled_coords) != 1000:
        try:
            t = np.linspace(0, 1, len(scaled_coords))
            t_new = np.linspace(0, 1, 1000)

            interp_func_x = interp1d(t, scaled_coords[:, 0], kind="linear")
            interp_func_y = interp1d(t, scaled_coords[:, 1], kind="linear")

            interpolated_coords = np.column_stack((interp_func_x(t_new), interp_func_y(t_new)))
        except Exception as e:
            print(f"Interpolation error: {e}")

    example["2D_Sequence_Interpolated"] = interpolated_coords.tolist()

    return example


ds_train = load_dataset("Hack90/experiment_one_viral_genomes_train_set")
ds_valid = load_dataset("Hack90/experiment_one_viral_genomes_val_set")
ds_test = load_dataset("Hack90/experiment_one_viral_genomes_test_set")

repo_ids = [
    "DNA-LLM/experiment_one_viral_genomes_train_set",
    "DNA-LLM/experiment_one_viral_genomes_val_set",
    "DNA-LLM/experiment_one_viral_genomes_test_set",
]

for repo_id, ds in zip(repo_ids, [ds_train, ds_valid, ds_test], strict=False):
    print(f"Processing {repo_id}")
    ds.map(generate_2d_sequence, num_proc=4)
    ds.push_to_hub(repo_id)
    print(f"Pushed {repo_id} to huggingface.co")
