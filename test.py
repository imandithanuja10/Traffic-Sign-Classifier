import torch
from models.generator import Generator
from models.cnn import TrafficCNN
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader, num_classes = get_dataloaders("TRAFFIC_SIGN_CLASSIFIER_DATASET/Train")

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

cnn_model = TrafficCNN(num_classes).to(device)
cnn_model.load_state_dict(torch.load("classifier.pth"))
cnn_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:

        imgs, labels = imgs.to(device), labels.to(device)

        low_res = torch.nn.functional.interpolate(imgs, scale_factor=0.5)
        enhanced = generator(low_res)

        outputs = cnn_model(enhanced)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", 100 * correct / total)