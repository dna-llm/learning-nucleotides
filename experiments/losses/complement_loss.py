import torch
from transformers import Trainer


class ComplementLoss(Trainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.logits_method = None

    def get_logits(self, model_output):
        # Determine and use the correct method to access logits
        if self.logits_method is None:
            try:
                _ = model_output[0]
                self.logits_method = "tuple"
            except TypeError:
                self.logits_method = "attribute"

        if self.logits_method == "tuple":
            return model_output[0]
        elif self.logits_method == "attribute":
            return model_output.logits

    def compute_loss(self, model, inputs):
        input_ids = inputs.pop("input_ids")
        labels = input_ids.clone().to('cuda')
        input_ids = input_ids[:, :-1].contiguous()  # Truncate the input_ids

        # Generate lm_logits for the original sequence
        # lm_logits = model(input_ids).logits
        lm_logits = self.get_logits(model(input_ids))
        shift_labels = labels[:, 1:].contiguous()

        # Compute standard loss
        loss_fct = torch.nn.CrossEntropyLoss()
        standard_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), shift_labels.view(-1))

        # Generate complement input_ids and lm_logits for the complement sequence
        complement_input_ids = self.get_complement_ids(input_ids)
        # complement_lm_logits = model(complement_input_ids).logits
        complement_lm_logits = self.get_logits(model(complement_input_ids))
        # Compute complement loss
        complement_labels = self.get_complement_labels(shift_labels)
        complement_loss = loss_fct(
            complement_lm_logits.view(-1, complement_lm_logits.size(-1)), complement_labels.view(-1)
        )

        # Sum the standard loss and complement loss
        total_loss = standard_lm_loss + complement_loss

        return total_loss

    @staticmethod
    def get_complement_ids(input_ids):
        complement_mapping = {3: 6, 4: 5, 5: 4, 6: 3}  # Mapping for A-T, C-G, G-C, T-A
        complement_ids = input_ids.clone()

        for key, value in complement_mapping.items():
            complement_ids[input_ids == key] = value

        return complement_ids

    @staticmethod
    def get_complement_labels(labels):
        complement_mapping = {3: 6, 4: 5, 5: 4, 6: 3}  # Mapping for A-T, C-G, G-C, T-A
        complement_labels = labels.clone()

        for key, value in complement_mapping.items():
            complement_labels[labels == key] = value

        return complement_labels
