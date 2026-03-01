import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Load one image from dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder("TRAFFIC_SIGN_CLASSIFIER_DATASET/Train", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

img, _ = next(iter(loader))
img = img.to(device)

# Create low resolution image
low_res = torch.nn.functional.interpolate(img, scale_factor=0.5)

# Generate enhanced image
with torch.no_grad():
    enhanced = generator(low_res)

# Convert tensors to displayable images
def imshow(tensor, title):
    image = tensor.cpu().squeeze().permute(1, 2, 0)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
imshow(img, "Original")

plt.subplot(1,3,2)
imshow(low_res, "Low Resolution")

plt.subplot(1,3,3)
imshow(enhanced, "Enhanced (SRGAN)")

plt.show()

from torchvision.utils import save_image

# Save enhanced image
save_image(enhanced, "enhanced.png", normalize=True, value_range=(-1, 1))
print("Enhanced image saved as enhanced.png")