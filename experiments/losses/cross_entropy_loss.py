import torch
from transformers import Trainer


class StandardLoss(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_method = None

    def get_logits(self, model_output):
        if self.logits_method is None:
            if isinstance(model_output, tuple):
                self.logits_method = "tuple"
            elif hasattr(model_output, "logits"):
                self.logits_method = "attribute"
            else:
                self.logits_method = "tensor"

        if self.logits_method == "tuple":
            return model_output[0]
        elif self.logits_method == "attribute":
            return model_output.logits
        else:
            return model_output

    def compute_loss(self, model, inputs):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", input_ids.clone())
        # Truncate input_ids and shift labels
        input_ids = input_ids[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_logits = model(input_ids)
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), shift_labels.view(-1))

        return lm_loss
