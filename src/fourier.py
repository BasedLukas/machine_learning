import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Fourier features
def fourier_features(x, B):
    x_proj = torch.cat([torch.sin(2 * 3.14159 * B * x), torch.cos(2 * 3.14159 * B * x)], dim=-1)
    return x_proj

# Create 2D grid
N = 50
coords = torch.linspace(0, 1, N)
x, y = torch.meshgrid(coords, coords)
coords = torch.stack([x.flatten(), y.flatten()], dim=-1)

# Fourier features
B = 5
coords_fourier = fourier_features(coords, B)

# Load and preprocess image
image_path = "image.png"
image = Image.open(image_path).convert('RGB').resize((N, N))  # Ensure the image is in RGB format
transform = transforms.ToTensor()
image_tensor = transform(image)
print(image_tensor.shape) 
image_tensor = image_tensor.permute(1, 2, 0).reshape(-1, 3)


# Neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = Model()

# Compile and train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100000):
    optimizer.zero_grad()
    outputs = model(coords_fourier)
    loss = criterion(outputs, image_tensor)
    loss.backward()
    optimizer.step()

# Generate image
image_pred = model(coords_fourier).detach().view(N, N, 3)
plt.imshow(image_pred)
plt.show()
