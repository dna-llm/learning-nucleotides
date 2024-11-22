import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import torch.linalg as la 

class CrossEntropyplusEnergy(Trainer):
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
        label_one_hot = F.one_hot(labels.long(), num_classes=8).float()
        
        mask = (labels > 2).float() * (labels < 7).float()
        mask_batch_size = mask.shape[0] 
        mask_o = mask
        mask =  mask.reshape(mask_batch_size * 2047,1)

        # CE Loss 
        loss_fct = torch.nn.CrossEntropyLoss()
        ce_loss = loss_fct(logits.reshape(mask_batch_size*2047,8) * mask, label_one_hot.reshape(mask_batch_size*2047,8) * mask).mean()
        # Energy Loss
        probabilities = F.softmax(logits, dim=-1)

        energy_matrix = torch.tensor([
            [ 0.0],
            [ 0.0],
            [ 0.0],
            [55.0],
            [45.0],
            [53.5],
            [45.0],
            [0.0]
        ]).to(device)

        pred_transformed = torch.matmul(probabilities, energy_matrix)
        label_one_hot = F.one_hot(labels.long(), num_classes=8).float()
        label_transformed = torch.matmul(label_one_hot, energy_matrix)
        label_transformed = label_transformed * (mask_o.reshape(mask_batch_size ,2047,1))
        pred_transformed = pred_transformed * (mask_o.reshape(mask_batch_size, 2047, 1) )
        label_cumsum = label_transformed.cumsum(2)
        pred_cumsum = pred_transformed.cumsum(2)

        #mask away irrelevant logits/labels
        label_cumsum = label_cumsum * (mask_o.reshape(mask_batch_size ,2047,1))
        pred_cumsum = pred_cumsum * (mask_o.reshape(mask_batch_size, 2047,1) )
        diff = pred_cumsum - label_cumsum
        
        energy_loss = torch.abs(diff).sum(2).mean()
        final_loss = ce_loss +  energy_loss

        return final_loss

