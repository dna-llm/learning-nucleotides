from datasets import load_dataset
from shutil import rmtree
from pathlib import Path

import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, StackDataset
from torch.optim import Adam

from einops import rearrange
from einops.layers.torch import Rearrange

from transfusion_pytorch import Transfusion, print_modality_sample
from tqdm import tqdm
import matplotlib.pyplot as plt


def divisible_by(num, den):
    return (num % den) == 0

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

# dataset


ds = load_dataset('DNA-LLM/generated_train_diff')
ds.set_format('torch')
means = ds['train']['seq'].float().mean()
stds = ds['train']['seq'].float().std()
normalized_data = (ds['train']['seq'].float() - means) / stds
dataset = StackDataset(ds['train']['integer_representation'].long(), normalized_data.float().reshape(168000,1,2048))

##### Mapping ####
data_labels= {1.7: 0, 1: 1, 1.6: 2, 1.8: 3, 2: 4, 3: 5, 4: 6, 6: 7, 
7: 8, 8: 9, 5: 10, 9: 11, 10: 12, 1.5: 13,
 20: 14, 30: 15, 40: 16, 50: 17, -1: 18}
inv_map = {v: k for k, v in data_labels.items()}

def decode_labelings(tensors):
    tensors = tensors.detach().flatten().cpu().numpy().tolist()
    tensors = [inv_map[i] for i in tensors ]
    return tensors 

# AE stuff
autoencoder_train_steps = 15_000
dim_latent = 16

encoder = nn.Sequential(
    # nn.nn.Conv1d(1, 1, kernel_size=1, stride=4),
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(128, dim_latent),
    Normalize()
).cuda()


decoder = nn.Sequential(
    nn.Linear(dim_latent,128),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(128,256),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(256,512),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(512,1024),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Linear(1024, 2048)
).cuda()


autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
autoencoder_dataloader = DataLoader(dataset, batch_size = 1024, shuffle = True)
autoencoder_iter_dl = cycle(autoencoder_dataloader)


print('training autoencoder')

with tqdm(total = 5) as pbar:
    for _ in range(8400):
        _, images = next(autoencoder_iter_dl)
        images = images.cuda()

        latents = encoder(images)
        latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.02) # add a bit of noise to latents
        reconstructed = decoder(latents)

        loss = F.mse_loss(images, reconstructed)

        loss.backward()

        pbar.set_description(f'loss: {loss.item():.5f}')

        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()

        pbar.update()


# encode the data 

autoencoder_dataloader = DataLoader(dataset, batch_size = 1024, shuffle = True)
encodings = []

print('encoding the data')


for batch in tqdm(autoencoder_dataloader):
    _, images = batch
    images = images.cuda()

    latents = encoder(images)
    encodings = encodings + latents.detach().cpu().numpy().tolist()

dataset = StackDataset(ds['train']['integer_representation'].long(), torch.Tensor(encodings))

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 100


model = Transfusion(
    num_text_tokens = 20,
    dim_latent = 16,
    modality_default_shape = (2,),  
    transformer = dict(
        dim = 512,
        depth = 8
    )
).cuda()


ema_model = model.create_ema(0.9)
dataloader = model.create_dataloader(dataset, batch_size = 1024, shuffle = True)
iter_dl = cycle(dataloader)
optimizer = Adam(model.parameters(), lr = 8e-4)

print('training model')
for step in range(1, 100_000 + 1):

    for _ in range(4):
        loss = model.forward(next(iter_dl))
        (loss / 4).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        sample = ema_model.sample()

        print_modality_sample(sample)

        if len(sample) < 1:
            continue

        text_tensor, maybe_image, *_ = sample

        if not isinstance(maybe_image, tuple):
            continue

        _, image = maybe_image
        image = decoder(image)

        text_tensor = text_tensor[text_tensor < 20] 

        text = decode_labelings(text_tensor)
        filename = str(results_folder / f'{step}.{text}.png')
        plt.plot(image.flatten().detach().cpu().numpy())
        plt.savefig(filename) 
