import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import get_dataloaders, set_seed, train_one_epoch, validate, save_model, setup_logger
from model_loader import load_model

def main(args):
    set_seed(2017)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_path = os.path.join("logs", f"{args.model_type}_{args.init_type}.log")
    
    logger = setup_logger(log_path)
    writer = SummaryWriter(log_dir=os.path.join("runs", f"{args.model_type}_{args.init_type}"))
    logger.info("Preparing data...")
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)

    logger.info("Loading model...")
    model = load_model(args.model_type, pretrained=(args.init_type == "pretrained"), num_classes=101).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.model_type == "resnet18":
        param_groups = [
            {'params': [param for name, param in model.named_parameters() if "fc" in name], 'lr': args.lr_fc},
            {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': args.lr_backbone}
        ]
    else:  # AlexNet
        param_groups = [
            {'params': [param for name, param in model.named_parameters() if "classifier.6" in name], 'lr': args.lr_fc},
            {'params': [param for name, param in model.named_parameters() if "classifier.6" not in name], 'lr': args.lr_backbone}
        ]

    optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    best_acc = 0.0
    print(f"Training {args.model_type} with {args.init_type} initialization")
    logger.info(f"Training {args.model_type} with {args.init_type} initialization")
    print(f"Batch size: {args.batch_size}")
    logger.info(f"Batch size: {args.batch_size}")
    print(f"Learning rate for fc: {args.lr_fc}")
    logger.info(f"Learning rate for fc: {args.lr_fc}")
    print(f"Learning rate for backbone: {args.lr_backbone}")
    logger.info(f"Learning rate for backbone: {args.lr_backbone}")
    # Add variables for early stopping
    early_stop_counter = 0
    prev_val_acc = 0

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        print(f"epoch {epoch+1}/{args.epochs}, train loss: {train_loss:.4f}")
        logger.info(f"Train loss: {train_loss:.4f}")
        val_acc = validate(model, val_loader, device, writer, epoch)
        print(f"epoch {epoch+1}/{args.epochs}, val acc: {val_acc:.4f}")
        logger.info(f"Validation accuracy: {val_acc:.4f}")

        # Early stopping check
        if epoch > 100:
            if val_acc < prev_val_acc:
                early_stop_counter += 1
                logger.info(f"Validation accuracy decreased ({prev_val_acc:.4f} -> {val_acc:.4f}). "
                          f"Early stopping counter: {early_stop_counter}/2")
                
                if early_stop_counter >= 2 and not args.tune_mode:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                early_stop_counter = 0
        
        prev_val_acc = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            if not args.tune_mode:  # Save the best model only if not in tuning mode
                save_path = os.path.join("checkpoints", f"{args.model_type}_{args.init_type}_best.pth")
                save_model(model, save_path)
                logger.info(f"Saved best model with accuracy: {val_acc:.4f}")

    writer.close()
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="caltech101_split")
    parser.add_argument("--model_type", type=str, choices=["resnet18", "alexnet"], default="resnet18")
    parser.add_argument("--init_type", type=str, choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_fc", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--tune_mode", action="store_true", help="When tuning, don't save models but keep logging")
    args = parser.parse_args()
    main(args)
