import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.cnn import TrafficCNN
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, num_classes = get_dataloaders("TRAFFIC_SIGN_CLASSIFIER_DATASET/Train")

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

cnn_model = TrafficCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    for imgs, labels in train_loader:

        imgs, labels = imgs.to(device), labels.to(device)

        low_res = nn.functional.interpolate(imgs, scale_factor=0.5)
        enhanced = generator(low_res)

        outputs = cnn_model(enhanced)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}  Loss: {loss.item()}")

torch.save(cnn_model.state_dict(), "classifier.pth")