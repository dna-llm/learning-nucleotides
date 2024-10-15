import pandas as pd
import torch
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from datasets import load_dataset
from tqdm import tqdm

model = DiffusionModel(
    net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
    in_channels=5,  # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 1, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=8,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
)

datasets = []
for n in range(6):
    ds = load_dataset(f"Hack90/five_d_{n}")
    df = pd.DataFrame(ds["train"])
    df["tensors"] = [torch.Tensor(df["tensors"][k]) for k in range(len(df))]
    datasets.append(df)


num_epochs = 10000
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

for epoch in tqdm(range(num_epochs)):
    for n in range(6):
        df = datasets[n]
        i = 0
        for _k in tqdm(range(len(df) // 8)):
            input_data = torch.cat(
                (
                    df.tensors.iloc[i],
                    df.tensors.iloc[i + 1],
                    df.tensors.iloc[i + 2],
                    df.tensors.iloc[i + 3],
                    df.tensors.iloc[i + 4],
                    df.tensors.iloc[i + 5],
                    df.tensors.iloc[i + 6],
                    df.tensors.iloc[i + 7],
                ),
                0,
            ).to(device)
            loss = model(input_data)
            i += 8
            if epoch % 100 == 0:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
