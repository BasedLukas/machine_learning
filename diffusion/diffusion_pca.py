from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.decomposition import PCA
from PIL import Image


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

def block(in_channels:int, out_channels:int): 
    """changes the channel number of the input"""
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 7, padding=3),
    nn.BatchNorm2d(out_channels),
    nn.LeakyReLU(),
    )

def compress_pca(image: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Compress an image using a pre-fitted PCA model.
    
    Parameters:
    - image: A 2D numpy array representing the grayscale image.
    - pca: Pre-fitted PCA model.
    
    Returns:
    - Compressed image representation.
    """
    # Flatten the image if it's not already
    image_flattened = image.flatten().reshape(1, -1)
    
    # Transform the image data to the principal components
    compressed_image = pca.transform(image_flattened)
    
    return compressed_image

def reconstruct_pca(compressed_image: np.ndarray, pca: PCA, original_shape: tuple) -> np.ndarray:
    """
    Reconstruct an image from its compressed PCA representation using a pre-fitted PCA model.
    
    Parameters:
    - compressed_image: The compressed image representation obtained from PCA.
    - pca: Pre-fitted PCA model.
    - original_shape: The original shape of the image before compression.
    
    Returns:
    - A 2D numpy array representing the reconstructed image.
    """
    # Reconstruct the image data from the principal components
    reconstructed_image = pca.inverse_transform(compressed_image)
    
    # Reshape the reconstructed image to its original shape
    reconstructed_image = reconstructed_image.reshape(original_shape)
    
    return reconstructed_image



class UnetStyleModel(nn.Module):
    """input is batch, n_channel, height, width"""

    def __init__(self, n_channel: int) -> None:
        super(UnetStyleModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            block(n_channel, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        unet_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.unet_model = unet_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using unet_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.unet_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.unet_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_mnist(n_epoch: int = 100, device="cuda:0", channels=1) -> None:

    ddpm = DDPM(unet_model=UnetStyleModel(channels), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    data_samples = []
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)


    # Flatten and stack images for PCA
    for images, _ in dataloader:  # Assuming dataloader is defined and loaded with MNIST dataset
        for img in images:
            flattened_img = img.numpy().flatten()
            data_samples.append(flattened_img)
    data_samples = np.stack(data_samples)
    # Fit PCA on the dataset
    n_components = 225 #allows int sqrt
    pca = PCA(n_components=n_components)
    pca.fit(data_samples)
    #create a dataloader of compressed images
    compressed_images = []
    for images, _ in dataloader:  # Assuming dataloader is defined and loaded with MNIST dataset
        for img in images:
            compressed_img = compress_pca(img.numpy(), pca)
            compressed_images.append(compressed_img)
    compressed_images = np.stack(compressed_images)
    compressed_dataloader = DataLoader(compressed_images, batch_size=256, shuffle=True, num_workers=0)


    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(compressed_dataloader)
        loss_ema = None
        for x in pbar:
            x = x[:,:,None,:]#add dim
            #reshape from (batch, 1,1,components) to (batch, 1,root(n_components) , root(n_components))
            x = x.reshape(x.shape[0], 1, int(np.sqrt(n_components)), int(np.sqrt(n_components)))
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
            samples = ddpm.sample(10, (1,1, n_components), device)
            for j, sampled in enumerate(samples):
                img = sampled[0,0,:].cpu().numpy()
                img = reconstruct_pca(img, pca, (28, 28))
                img = Image.fromarray(img).convert("L")
                img.save(f"./contents/ddpm_sample_{i}_{j}.png")


            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")
            # save_image(img, f"./contents/ddpm_sample_{i}.png")




if __name__ == "__main__":
    train_mnist(5)
    
