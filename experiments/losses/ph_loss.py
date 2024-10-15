import logging
from typing import ClassVar

import torch
from torch import nn

# from topologylayer.nn import AlphaLayer
from transformers import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistentHomologyLayer:
    NUCLEOTIDE_MAPPING: ClassVar[dict[int, torch.Tensor]] = {
        3: torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        4: torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        5: torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        6: torch.tensor([0, 0, 0, 1], dtype=torch.float32),
    }

    @classmethod
    def chaos_4d_representation(cls, dna_sequence_tensor, sample_rate=10):
        def encode_nucleotide_to_vector(nucleotide):
            if nucleotide.numel() > 1:
                return torch.stack([
                    cls.NUCLEOTIDE_MAPPING.get(
                        n.item(), torch.tensor([0, 0, 0, 0], dtype=torch.float32)
                    )
                    for n in nucleotide
                ])
            else:
                return cls.NUCLEOTIDE_MAPPING.get(
                    nucleotide.item(), torch.tensor([0, 0, 0, 0], dtype=torch.float32)
                )

        points = [encode_nucleotide_to_vector(dna_sequence_tensor[0])]
        points[0] = (
            points[0] if points[0] is not None else torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        )

        for nucleotide in dna_sequence_tensor[1:]:
            vector = encode_nucleotide_to_vector(nucleotide)
            next_point = 0.5 * (points[-1] + vector)
            points.append(next_point)

        points_tensor = torch.stack(points).requires_grad_(True)
        unique_points, _ = torch.unique(points_tensor, dim=0, return_inverse=True)

        # sampled_indices = torch.linspace(
        #     0, len(points_tensor) - 1, steps=sample_rate, dtype=torch.long
        # )
        # sampled_points = points_tensor[sampled_indices]

        return points_tensor


class PHLoss(Trainer):
    def compute_loss(self, model, inputs):
        input_ids = inputs.pop("input_ids")
        labels = input_ids.clone().to(model.device)
        input_ids = input_ids[:, :-1].contiguous()  # Truncate the input_ids
        lm_logits = model(input_ids).logits
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), shift_labels.view(-1))

        layer = nn.Linear()  # AlphaLayer(maxdim=2)
        logger.info(input_ids)
        logger.info(f"Input IDs shape: {input_ids.shape}")
        logger.info(f"Input IDs: {input_ids}")

        labels_chaos = self.get_chaos_representation(input_ids.flatten())
        generated_labels = lm_logits.argmax(dim=-1)
        try:
            generated_chaos = self.get_chaos_representation(generated_labels.flatten())
            logger.info(f"Chaos shape: {generated_labels.shape}")
            logger.info(f"Chaos labels: {generated_labels}")
            logger.info(f"generated Chaos: {generated_chaos}")
        except (TypeError, ValueError, IndexError):
            generated_chaos = [0]
        if len(generated_chaos) > 800:
            try:
                # dgms_labels, _ = layer(labels_chaos)
                # dgms_generated, _ = layer(generated_chaos)
                dgms_labels, _ = layer(labels_chaos.to(model.device))
                dgms_generated, _ = layer(generated_chaos.to(model.device))
                logger.info(f"DGMS: {dgms_generated}")

                # Distance of the second diagram of each of the diagrams
                distance = torch.cdist(dgms_generated[1], dgms_labels[1])

                total_loss = lm_loss + distance
                logger.info(
                    f"LM Loss: {lm_loss.item()}, Distance: {distance.item()}, Total Loss: {total_loss.item()}"
                )
                return total_loss
            except Exception as e:
                logger.info(f"Failure caused by:{e}")
                return lm_loss
        else:
            return lm_loss

    @staticmethod
    def get_chaos_representation(input_ids, drop_invalid=True, max_repeats=5):
        if drop_invalid:
            valid_nucleotides = torch.tensor([3, 4, 5, 6], device=input_ids.device)
            mask = torch.isin(input_ids, valid_nucleotides)
            filtered_input_ids = input_ids[mask]
            logger.info(f"filtered_input:{filtered_input_ids}")
        else:
            filtered_input_ids = input_ids

        # Find consecutive repeats of the same nucleotide
        diff = filtered_input_ids[1:] != filtered_input_ids[:-1]
        diff = torch.cat((torch.tensor([True], device=diff.device), diff))
        start_indices = torch.where(diff)[0]
        end_indices = torch.cat((
            start_indices[1:],
            torch.tensor([len(filtered_input_ids)], device=start_indices.device),
        ))
        repeat_lengths = end_indices - start_indices

        # Filter out repeats that occur more than max_repeats times
        valid_repeat_mask = repeat_lengths <= max_repeats
        valid_start_indices = start_indices[valid_repeat_mask]
        valid_end_indices = end_indices[valid_repeat_mask]

        valid_input_ids = torch.cat([
            filtered_input_ids[start:end]
            for start, end in zip(valid_start_indices, valid_end_indices, strict=False)
        ])

        return PersistentHomologyLayer.chaos_4d_representation(valid_input_ids)
