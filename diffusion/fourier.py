
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from PIL import Image

def compress(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compress an image using the Fourier Transform by removing frequencies with magnitudes below a threshold.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - threshold: Magnitude threshold for frequency components.
    
    Returns:
    - A numpy array containing the compressed frequency representation of the image.
    """
    # Apply Fast Fourier Transform
    fft_image = np.fft.fft2(image)
    # Shift the zero frequency component to the center of the spectrum
    fft_shift = np.fft.fftshift(fft_image)
    
    # Apply thresholding
    magnitude_spectrum = np.abs(fft_shift)
    fft_shift[magnitude_spectrum < threshold] = 0
    
    # Return the compressed frequency representation
    return fft_shift

def expand(compressed_image: np.ndarray) -> np.ndarray:
    """
    Expand a compressed image back to the spatial domain using the Inverse Fast Fourier Transform.
    
    Parameters:
    - compressed_image: Compressed frequency representation of the image.
    
    Returns:
    - A 2D numpy array representing the expanded image.
    """
    # Shift the zero frequency component back to the original position
    ifft_shift = np.fft.ifftshift(compressed_image)
    # Apply Inverse Fast Fourier Transform
    img_expanded = np.fft.ifft2(ifft_shift)
    # Take the real part to get rid of any imaginary part resulted from the transformation
    return np.real(img_expanded)

class FourierCompress(object):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, img):
        # Convert PIL image to NumPy array
        img_np = np.array(img)
        
        # Apply FFT
        fft_img = np.fft.fft2(img_np)
        fft_shift = np.fft.fftshift(fft_img)
        
        # Thresholding
        magnitude_spectrum = np.abs(fft_shift)
        fft_shift[magnitude_spectrum < self.threshold] = 0
        

        # IFFT to reconstruct the image
        ifft_shift = np.fft.ifftshift(fft_shift)
        img_reconstructed = np.fft.ifft2(ifft_shift)
        img_reconstructed = np.real(img_reconstructed)
        
        # Convert back to PIL Image to be compatible with further PyTorch transformations
        img_reconstructed = Image.fromarray(img_reconstructed.astype(np.uint8))
        print()
        return img_reconstructed


###
def compress_pca(image: np.ndarray, n_components: int) -> tuple:
    """
    Compress an image using PCA, reducing it to the specified number of principal components.
    
    Parameters:
    - image: A 2D numpy array representing the grayscale image.
    - n_components: The number of principal components to keep.
    
    Returns:
    - A tuple containing the compressed image representation, the PCA components, and the mean of the original image.
    """
    # Flatten the image to a 2D array if it's not already
    if image.ndim == 3:
        # Assuming image is in HxWxC format, where C is the color channels
        image = image.reshape(image.shape[0], -1)
    else:
        image = image.reshape(1, -1)

    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform the image data to the principal components
    compressed_image = pca.fit_transform(image)
    
    # Return the compressed representation, PCA components, and mean of the original data
    return (compressed_image, pca.components_, pca.mean_, n_components)

def reconstruct_pca(compressed_image: np.ndarray, pca_components: np.ndarray, mean: np.ndarray, n_components: int, original_shape: tuple) -> np.ndarray:
    """
    Reconstruct an image from its compressed PCA representation.
    
    Parameters:
    - compressed_image: The compressed image representation obtained from PCA.
    - pca_components: The principal components from PCA.
    - mean: The mean of the original image.
    - n_components: The number of principal components used in the compression.
    - original_shape: The original shape of the image before compression.
    
    Returns:
    - A 2D numpy array representing the reconstructed image.
    """
    # Reconstruct the image data from the principal components
    reconstructed_image = np.dot(compressed_image, pca_components[:n_components]) + mean
    
    # Reshape the reconstructed image to its original shape
    reconstructed_image = reconstructed_image.reshape(original_shape)
    
    return reconstructed_image

###


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

# Process and display the first batch
for images, _ in dataloader:
    for i, img in enumerate(images):

        img = img[0].numpy()
        plt.imshow(img, cmap="gray")
        plt.show()

        (compressed_image, pcacomponents_, pcamean_, n_components) = compress_pca(img, 10)
        reconstructed_image = reconstruct_pca(compressed_image, pcacomponents_, pcamean_, n_components, img.shape)
        plt.imshow(img, cmap="gray")
        plt.show()
        break
        
    break


