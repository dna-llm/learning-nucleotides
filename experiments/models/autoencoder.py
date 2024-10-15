import torch
from torch import Tensor, nn


def build_model(
    linear_sizes: list[int],
    activation: str,
    sequence_length: int,
    is_decoder: bool = False,
) -> nn.Sequential:
    model = nn.Sequential()
    if activation.lower() == "leakyrelu":
        activation_fn = nn.LeakyReLU
    else:
        raise ValueError("Please specify a valid activation function.")

    input_size = sequence_length if not is_decoder else linear_sizes[-1]
    linear_list = linear_sizes if not is_decoder else list(reversed(linear_sizes[:-1]))
    for size in linear_list:
        model.add_module(f"linear_{input_size}_{size}", nn.Linear(input_size, size))
        model.add_module(f"activation_{size}", activation_fn())
        input_size = size

    if is_decoder:
        output_size = sequence_length
        model.add_module(f"linear_{input_size}_{output_size}", nn.Linear(input_size, output_size))

    return model


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = nn.LeakyReLU
        # build encoder
        self.encoder = nn.Sequential()
        input_size = self.input_dim
        for layer, dim_size in enumerate(hidden_sizes):
            self.encoder.add_module(f"linear_{layer}", nn.Linear(input_size, dim_size))
            self.encoder.add_module(f"LeakyReLU_{layer}", self.activation())
            input_size = dim_size
        #  build decoder
        self.decoder = nn.Sequential()
        input_size = self.latent_dim
        for layer, dim_size in enumerate(hidden_sizes[::-1]):
            self.decoder.add_module(f"linear_{layer}", nn.Linear(input_size, dim_size))
            self.decoder.add_module(f"LeakyReLU_{layer}", self.activation())
            input_size = dim_size
        # final output layer
        self.decoder.add_module(
            f"linear_{dim_size}_{self.input_dim}", nn.Linear(dim_size, self.input_dim)
        )
        # learnable parameters for latent space
        self.fc_mu = nn.Linear(hidden_sizes[-1], self.latent_dim)
        self.fc_log_var = nn.Linear(hidden_sizes[-1], self.latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
