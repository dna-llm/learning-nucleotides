import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import torch.linalg as la 

class TwoDRepL2CELoss(Trainer):
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
        ]).to(device)
        
        pred_transformed = torch.matmul(probabilities, transform_matrix)
        label_one_hot = F.one_hot(labels.long(), num_classes=8).float()
        label_transformed = torch.matmul(label_one_hot, transform_matrix)
        
        pred_cumsum = pred_transformed.cumsum(2)
        label_cumsum = label_transformed.cumsum(2)

        mask = mask.reshape(2, 2047,1)
        label_cumsum = label_cumsum * mask
        pred_cumsum = pred_cumsum * mask 
        diff = pred_cumsum - label_cumsum

        
        geometric_loss = la.norm(diff, ord=2, axis=0).mean()  
        
        loss_fct = torch.nn.CrossEntropyLoss()
        
        logits = logits * mask
        label_one_hot = label_one_hot * mask

        lm_loss = loss_fct(logits, label_one_hot)
        
        final_loss = lm_loss + geometric_loss
        return final_loss
