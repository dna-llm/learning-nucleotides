import torch
from transformers import Trainer
import torch.nn.functional as F

class Discrete_CE_Loss(Trainer):
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

        x= inputs['input_ids']
        should_noise = inputs['attention_mask']
        scheduler = torch.linspace(
            1 / 2048, 1, steps=2048, dtype=torch.float32, device=x.device
        )  
        t = torch.randint(0, 2024, [x.size(0)], device=x.device)
        t=t.unsqueeze(1)
        mask_prob = scheduler[t].expand(-1, x.shape[1])
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)
      
        if should_noise is not None:
            will_mask &= should_noise.bool()

        noised_x = x.clone()
        noised_x[will_mask] = torch.Tensor([2]).long()
      
        logits =model(noised_x, t.flatten())  

        target = x.clone()

        target[noised_x != 2] = -100

        loss = F.cross_entropy(
            input=logits.transpose(-1, -2),
            target=target,
            reduction="mean",
        )

        return loss
