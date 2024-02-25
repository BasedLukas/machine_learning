from sklearn.decomposition import PCA
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Assuming dataset is already loaded and is a list of numpy arrays
data_samples = []  # This should be filled with flattened images from the dataset

tf = transforms.Compose(
    [   
        # FourierCompress(50),
        # transforms.Pad(2),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0))
    ]
)

dataset = MNIST(
    "./data",
    train=True,
    download=True,
    transform=tf,
)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0)

# Flatten and stack images for PCA
for images, _ in dataloader:  # Assuming dataloader is defined and loaded with MNIST dataset
    for img in images:
        flattened_img = img.numpy().flatten()
        data_samples.append(flattened_img)


data_samples = np.stack(data_samples)

# Fit PCA on the dataset
n_components = 256
pca = PCA(n_components=n_components)
pca.fit(data_samples)


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


# Example of processing an image
for images, _ in dataloader:
    img = images[0].numpy()  # Get the first image in the batch
    original_shape = img.shape  # Save the original shape for reconstruction
    
    compressed_image = compress_pca(img, pca)
    reconstructed_image = reconstruct_pca(compressed_image, pca, original_shape)
    
    # Display original and reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(img), cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(reconstructed_image), cmap="gray")
    plt.title("Reconstructed Image")
    plt.show()
    
     # Process only the first image for demonstration
