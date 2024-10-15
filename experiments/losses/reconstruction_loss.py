import torch
from transformers import Trainer


class VAELoss(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_loss(self, model, inputs):
        loss_function = torch.nn.MSELoss()
        wave = inputs.pop("2D_Sequence_Interpolated")[:, :, 1].to(self.device)
        reconstructed, mu, log_var = model(wave)
        mse_loss = loss_function(reconstructed, wave)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = mse_loss + kld_loss

        return loss
