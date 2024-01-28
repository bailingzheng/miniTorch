import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models import ResNet, MobileNetV2
from optim import AdamW


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        y_pred = model(x)

        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"train epoch: {epoch} [{batch_idx * len(x)} / {len(train_loader.dataset)}] | loss: {loss.item():.4f}")


def test(model, test_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in test_loader:
           
            y_pred = model(x)

            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), ignore_index=-1)
            losses.append(loss.item())
            
        mean_loss = torch.tensor(losses).mean().item()

    print(f"test set: [{len(test_loader.dataset)}] | loss: {mean_loss:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MNIST Example")
    
    parser.add_argument(
        "--type", 
        type=str, 
        default="resnet", 
        help="model class type to use, resnet | mobilenetv2 "
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64, 
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", 
        type=int, 
        default=1000,
        help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-2,
        help="learning rate (default: 1e-2)"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.7, 
        help="Learning rate step gamma (default: 0.7)"
    )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    args = parser.parse_args()
    torch.manual_seed(3407)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}

    dataset1 = datasets.MNIST("examples/classification_models", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("examples/classification_models", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.type == "resnet":
        model = ResNet([1, 1, 1, 1], num_classes=10)
    elif args.type == "mobilenetv2":
        model = MobileNetV2(num_classes=10)
    else:
        raise ValueError(f"model type {args.type} is not recognized.")

    print(f"model #params: {sum(p.numel() for p in model.parameters()) / 1e6} M")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()