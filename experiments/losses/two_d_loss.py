import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import torch.linalg as la 

class TwoDRepLoss(Trainer):
    """
    TwoDRepLoss is a loss function for 2D representations of nucleotide sequences.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.logits_method = None

    def get_logits(self, model_output):
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
        device = "cuda"

        input_ids = inputs.pop("input_ids")
        logits = self.get_logits(model(input_ids[:, :-1]))
        labels = input_ids[:, 1:]

        # Use softmax probabilities
        probabilities = F.softmax(logits, dim=-1)
        mask = (labels > 2).float() * (labels < 7).float()

        transform_matrix = torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, -0.8660254],
            [0.8660254, 0.5],
            [0.8660254, -0.5],
            [0.5, 0.8660254],
            [0.0, 0.0]
        ])
        pred_transformed = torch.matmul(torch.nn.functional.softmax(output), transform_matrix)
        
        label_transformed = torch.matmul(torch.nn.functional.softmax(output), transform_matrix)
        
        pred_cumsum = torch.cumsum(pred_transformed, dim=1)
        label_cumsum = torch.cumsum(label_transformed, dim=1)
        
        diff = pred_cumsum - label_cumsum
        mask = torch.unsqueeze(mask, -1)
        diff = diff * mask   
        
        geometric_loss = diff.sum(1).mean(0).sum()

        return geometric_loss
