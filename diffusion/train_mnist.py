from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from original.mindiffusion.unet import NaiveUnet
from original.mindiffusion.ddpm import DDPM


def train_mnist(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(1, 1, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_mnist.pth"))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(),transforms.Pad(2), transforms.Normalize((0.5,), (1.0))]
    )# padding to 32x32 because of the size of the cifar dataset this was made for

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)
    times = []
    for i in range(n_epoch):
        times.append([time.time()])
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 32, 32), device)
            grid = make_grid(xh,nrow=4)
            save_image(grid, f"./contents/ddpm_sample_mnist{i}.png")
            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

        times[-1].append(time.time())

    return times

if __name__ == "__main__":
    times = train_mnist(100)
    total = 0
    for i in range(len(times)):
        print(f"Epoch {i} : {times[i][1]-times[i][0]}")
        total += times[i][1]-times[i][0]
    print(f"Total time: {total}")