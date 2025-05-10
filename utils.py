import os
import random
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def get_dataloaders(data_dir, batch_size=32, test_only=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transform

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if test_only:
        return None, None, test_loader
    return train_loader, val_loader, train_dataset.classes

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # store the loss for each step
        writer.add_scalar("Loss/train_step", loss.item(), epoch * len(loader) + step)

    avg_loss = running_loss / len(loader)
    # store the average loss for the epoch
    writer.add_scalar("Loss/train", avg_loss, epoch)

    return avg_loss

def validate(model, loader, device, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

    acc = correct / total
    avg_loss = running_loss / len(loader)

    # store the loss for each epoch
    writer.add_scalar("Loss/val", avg_loss, epoch)
    # store the accuracy for the epoch
    writer.add_scalar("Accuracy/val", acc, epoch)

    return acc


def save_model(model, path):
    torch.save(model.state_dict(), path)

def setup_logger(log_path):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        level=logging.INFO
    )
    return logging.getLogger()
