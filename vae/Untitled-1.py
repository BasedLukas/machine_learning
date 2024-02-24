# %%
import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from encoder import VAE_Encoder
from decoder import VAE_Decoder


# %%
# latent_dim = 5
batch_size = 1
epochs = 1
update_freq = 100 #weight uypdates after freq iterations
beta = 0.001

# %%
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        out = self.decoder(z)
        return out, mean, log_var

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2,batch_size=batch_size, shuffle=True)    

def imshow(img: torch.Tensor):
    """
    Displays an MNIST image.

    Parameters:
    img (torch.Tensor): A PyTorch tensor of the image to display. 
                        Expected shape is (1, 28, 28) for a single image.
    """
    # Check if the image tensor seems to be in the (C, H, W) format and has 1 channel (grayscale)
    if img.shape[0] == 1:
        img = img.squeeze(0)  # Remove the channel dimension if it's a single-channel image
    plt.imshow(img, cmap='gray')  # Display the image in grayscale
    plt.axis('off')  # Optional: Do not display axis for cleaner visualization
    plt.show()

# %%
# x = torch.randn(2, 1, 28, 28)
# model = VAE()
# x, mean, log_var = model(x)
# mean.shape, log_var.shape, x.shape   

# %%
device = 'cpu'# torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, verbose=True)
mse = nn.MSELoss()
def kl_divergence_loss(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

loss_history = []
update = 10

# %%
for epoch in range(epochs):
    model.train()
    print(f'Epoch {epoch+1}')
    for img, label in train_loader:
        img = img.to(device)
        output, mean, log_var = model(img)
        loss = mse(output, img)
        divergence = kl_divergence_loss(mean, log_var) * beta
        total_loss = loss + divergence
        total_loss.backward()

        if update % update_freq == 0:
            print(f'Loss: {loss.item():.4f}, divergence: {divergence.item():.4f}, total: {total_loss.item():.4f}')
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()
        loss_history.append(loss.item())
        update += 1




plt.plot(loss_history)
plt.show()
imshow(img[0].cpu())
imshow(output[0].cpu().detach())

# %%
def sample_vae(model, num_samples=1, latent_dim=4):
    model.eval()
    with torch.no_grad():
        # Sample from a standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        # Decode the sampled latent vectors
        generated_images = model.decoder(z)
    return generated_images
imshow(sample_vae(model, num_samples=1)[0].cpu().detach())

# %%
for img, label in test_loader:
    img = img.to(device)
    output, m, v = model(img)
    imshow(img[0].cpu())
    imshow(output[0].cpu().detach())
    print(label[0])
    break

# %%
def visualize_latent_space(encoder, data_loader, num_samples=1000):
    encoder.eval()
    samples, labels = next(iter(data_loader))
    with torch.no_grad():
        encoded, _, _ = encoder(samples.to(device)[:num_samples])
    encoded = encoded.cpu().numpy()
    labels = labels.numpy()[:num_samples]
    
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded)
    
    plt.figure(figsize=(10, 6))
    for i in range(10):  # Assuming 10 classes
        idxs = labels == i
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=str(i))
    plt.legend()
    plt.show()

# Example usage with your VAE encoder and a data loader
visualize_latent_space(model.encoder, test_loader)


