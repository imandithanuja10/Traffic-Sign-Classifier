import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _, _ = get_dataloaders("TRAFFIC_SIGN_CLASSIFIER_DATASET/Train")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

EPOCHS = 5

for epoch in range(EPOCHS):
    for imgs, _ in train_loader:

        imgs = imgs.to(device)
        low_res = nn.functional.interpolate(imgs, scale_factor=0.5)

        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # Train Discriminator
        fake_imgs = generator(low_res)
        d_real = discriminator(imgs)
        d_fake = discriminator(fake_imgs.detach())

        d_loss = criterion(d_real, real_labels) + \
                 criterion(d_fake, fake_labels)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        d_fake = discriminator(fake_imgs)
        g_loss = criterion(d_fake, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch+1}  D Loss: {d_loss.item()}  G Loss: {g_loss.item()}")

torch.save(generator.state_dict(), "generator.pth")