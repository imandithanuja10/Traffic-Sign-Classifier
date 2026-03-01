from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from preprocessing import get_train_transforms, get_test_transforms

BATCH_SIZE = 32

def get_dataloaders(dataset_path):

    train_dataset = ImageFolder(
        dataset_path,
        transform=get_train_transforms()
    )

    test_dataset = ImageFolder(
        dataset_path,
        transform=get_test_transforms()
    )

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, len(train_dataset.dataset.classes)