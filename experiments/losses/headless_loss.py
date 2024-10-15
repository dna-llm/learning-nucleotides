import torch
from transformers import Trainer


class Headless(Trainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.logits_method = None
        self.loss_fct = torch.nn.CrossEntropyLoss()

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
        model = model.module if hasattr(model, "module") else model
        input_ids = inputs.pop("input_ids")
        # Create teacherless input by replacing the target sequence with a special token
        teacherless_input = torch.full_like(input_ids, model.config.pad_token_id or 0)
        bos_token_id = model.config.bos_token_id if model.config.bos_token_id is not None else 1
        teacherless_input[:, 0] = bos_token_id
        labels = input_ids.to(model.device)
        labels[:, 0] = model.config.pad_token_id or 0
        # Compute the logits and loss
        teacherless_logits = self.get_logits(model(teacherless_input))
        lm_loss = self.loss_fct(
            teacherless_logits.view(-1, teacherless_logits.size(-1)), labels.view(-1)
        )

        return lm_loss
