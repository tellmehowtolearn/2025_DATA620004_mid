import torch
import torch.nn as nn
from utils import get_dataloaders, set_seed
from model_loader import load_model

def test_model(model_type, checkpoint_path, data_dir="caltech101_split", batch_size=32):
    set_seed(2017)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(data_dir, batch_size, test_only=True)

    model = load_model(model_type, pretrained=False, num_classes=101).to(device)

    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["resnet18", "alexnet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="caltech101_split")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    test_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
