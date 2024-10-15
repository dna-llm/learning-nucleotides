import math

import pywt
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class MultiresLayer(nn.Module):
    def __init__(
        self,
        d_model,
        kernel_size=None,
        depth=None,
        wavelet_init=None,
        tree_select="fading",
        seq_len=None,
        dropout=0.0,
        memory_size=None,
        indep_res_init=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.tree_select = tree_select
        if depth is not None:
            self.depth = depth
        elif seq_len is not None:
            self.depth = self.max_depth(seq_len)
        else:
            raise ValueError("Either depth or seq_len must be provided.")

        if tree_select == "fading":
            self.m = self.depth + 1
        elif memory_size is not None:
            self.m = memory_size
        else:
            raise ValueError("memory_size must be provided when tree_select != 'fading'")

        with torch.no_grad():
            if wavelet_init is not None:
                self.wavelet = pywt.Wavelet(wavelet_init)
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
                self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [d_model, 1, 1]))
                self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [d_model, 1, 1]))
            elif kernel_size is not None:
                self.h0 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1.0, 1.0)
                    * math.sqrt(2.0 / (kernel_size * 2))
                )
                self.h1 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1.0, 1.0)
                    * math.sqrt(2.0 / (kernel_size * 2))
                )
            else:
                raise ValueError("kernel_size must be specified for non-wavelet initialization.")

            w_init = torch.empty(d_model, self.m + 1).uniform_(-1.0, 1.0) * math.sqrt(
                2.0 / (2 * self.m + 2)
            )
            if indep_res_init:
                w_init[:, -1] = torch.empty(d_model).uniform_(-1.0, 1.0)
            self.w = nn.Parameter(w_init)

        self.activation = nn.GELU()
        dropout_fn = nn.Dropout1d
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

    def max_depth(self, L):
        depth = math.ceil(math.log2((L - 1) / (self.kernel_size - 1) + 1))
        return depth

    def forward(self, x):
        if self.tree_select == "fading":
            y = forward_fading(x, self.h0, self.h1, self.w, self.depth, self.kernel_size)
        elif self.tree_select == "uniform":
            y = forward_uniform(x, self.h0, self.h1, self.w, self.depth, self.kernel_size, self.m)
        else:
            raise NotImplementedError()
        y = self.dropout(self.activation(y))
        return y


def forward_fading(x, h0, h1, w, depth, kernel_size):
    res_lo = x
    y = 0.0
    dilation = 1
    L = x.shape[-1]
    for i in range(depth, 0, -1):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        res_hi = torch.nn.functional.conv1d(res_lo_pad, h1, dilation=dilation, groups=x.shape[1])
        res_lo = torch.nn.functional.conv1d(res_lo_pad, h0, dilation=dilation, groups=x.shape[1])

        # Trim res_hi and res_lo to match the input length L
        if res_hi.shape[-1] > L:
            res_hi = res_hi[..., -L:]
        if res_lo.shape[-1] > L:
            res_lo = res_lo[..., -L:]

        y += w[:, i : i + 1] * res_hi
        dilation *= 2

    y += w[:, :1] * res_lo
    y += x * w[:, -1:]
    return y


def forward_uniform(x, h0, h1, w, depth, kernel_size, memory_size):
    # x: [bs, d_model, L]
    coeff_lst = []
    dilation_lst = [1]
    dilation = 1
    res_lo = x
    for _ in range(depth):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        res_hi = torch.nn.functional.conv1d(res_lo_pad, h1, dilation=dilation, groups=x.shape[1])
        res_lo = torch.nn.functional.conv1d(res_lo_pad, h0, dilation=dilation, groups=x.shape[1])
        coeff_lst.append(res_hi)
        dilation *= 2
        dilation_lst.append(dilation)
    coeff_lst.append(res_lo)
    coeff_lst = coeff_lst[::-1]
    dilation_lst = dilation_lst[::-1]

    # y: [bs, d_model, L]
    y = uniform_tree_select(coeff_lst, dilation_lst, w, kernel_size, memory_size)
    y += x * w[:, -1:]
    return y


def uniform_tree_select(coeff_lst, dilation_lst, w, kernel_size, memory_size):
    latent_dim = 1
    y_lst = [coeff_lst[0] * w[:, 0, None]]
    layer_dim = 1
    dilation_lst[0] = 1
    for layer, coeff_l in enumerate(coeff_lst[1:]):
        if latent_dim + layer_dim > memory_size:
            layer_dim = memory_size - latent_dim
        # layer_w: [d, layer_dim]
        layer_w = w[:, latent_dim : latent_dim + layer_dim]
        # coeff_l_pad: [bs, d, L + left_pad]
        left_pad = (layer_dim - 1) * dilation_lst[layer]
        coeff_l_pad = torch.nn.functional.pad(coeff_l, (left_pad, 0), "constant", 0)
        # y: [bs, d, L]
        y = torch.nn.functional.conv1d(
            coeff_l_pad,
            torch.flip(layer_w[:, None, :], (-1,)),
            dilation=dilation_lst[layer],
            groups=coeff_l.shape[1],
        )
        y_lst.append(y)
        latent_dim += layer_dim
        if latent_dim >= memory_size:
            break
        layer_dim = 2 * (layer_dim - 1) + kernel_size
    return sum(y_lst)


def apply_norm(x, norm, batch_norm=False):
    if batch_norm:
        return norm(x)
    else:
        return norm(x.transpose(-1, -2)).transpose(-1, -2)


class MultiresTransformerConfig(PretrainedConfig):
    model_type = "multires_transformer"

    def __init__(
        self,
        n_tokens=8,
        d_model=128,
        n_layers=6,
        kernel_size=2,
        depth=4,
        dropout=0.1,
        d_mem=1024,
        indep_res_init=True,
        tree_select="fading",
        hinit=None,
        max_seqlen=1000,
        d_input=6,
        nr_logistic_mix=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout = dropout
        self.d_mem = d_mem
        self.indep_res_init = indep_res_init
        self.tree_select = tree_select
        self.hinit = hinit
        self.max_length = max_seqlen
        self.d_input = d_input
        self.nr_logistic_mix = nr_logistic_mix


class MultiresTransformer(PreTrainedModel):
    config_class = MultiresTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = nn.Embedding(config.n_tokens, config.d_model)
        self.seq_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.n_layers):
            layer = MultiresLayer(
                config.d_model,
                kernel_size=config.kernel_size,
                depth=config.depth,
                wavelet_init=config.hinit,
                tree_select=config.tree_select,
                seq_len=config.max_length,
                dropout=config.dropout,
                memory_size=config.d_mem,
                indep_res_init=config.indep_res_init,
            )
            self.seq_layers.append(layer)

            activation_scaling = 2
            mixing_layer = nn.Sequential(
                nn.Conv1d(config.d_model, activation_scaling * config.d_model, 1),
                nn.GLU(dim=-2),
                nn.Dropout1d(config.dropout),
                nn.Conv1d(config.d_model, config.d_model, 1),
            )
            self.mixing_layers.append(mixing_layer)
            self.norms.append(nn.LayerNorm(config.d_model))

        self.decoder = nn.Conv1d(config.d_model, config.n_tokens, 1)

        self.init_weights()

    def forward(self, input_ids):
        x = self.encoder(input_ids).transpose(1, 2)
        for layer, mixing_layer, norm in zip(
            self.seq_layers, self.mixing_layers, self.norms, strict=False
        ):
            x_orig = x
            x = layer(x)
            x = mixing_layer(x)
            x += x_orig
            x = apply_norm(x, norm)

        logits = self.decoder(x)
        # output: (batch_size, seq_len, vocab_size)
        return logits.transpose(1, 2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
