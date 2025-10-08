#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.models.resnet

# ---------------------------
# Utilities
# ---------------------------

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res

def get_data_loaders(data_dir, batch_size, workers):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds   = ImageFolder(val_dir,   transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader

def build_optimizer(params, opt_name, lr, weight_decay, betas):
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Use 'adam' or 'adamw'.")

# ---------------------------
# Training / Evaluation
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=50):
    model.train()
    running_loss = 0.0
    n = 0
    start = time.time()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            print(f"  [train] step {i+1}/{len(loader)}  "
                  f"loss {running_loss / n:.4f}  "
                  f"ips {n / elapsed:.1f} img/s")

    return running_loss / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    top1_sum = 0.0
    top5_sum = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs

        t1, t5 = accuracy(outputs, targets, topk=(1, 5))
        top1_sum += t1 * bs
        top5_sum += t5 * bs

    return total_loss / n, top1_sum / n, top5_sum / n

# ---------------------------
# Main
# ---------------------------

class ZeroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(grad_output)

        return grad_x

def Bottleneck_forward(self, x):
    identity = x

    x = ZeroGrad.apply(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(identity)

    out += identity
    out = self.relu(out)

    return out

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet with Adam/AdamW + TensorBoard")
    parser.add_argument("--data", type=str, required=True, help="Path to ImageNet root (expects train/ and val/)")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs (torchvision reference is 90)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3)  # <- default 0.001
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoints")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate a checkpoint (--resume)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume/evaluate")
    parser.add_argument("--zerograd", action="store_true", help="Zero gradients in residual blocks")
    parser.add_argument("--tag", type=str, default="", help="Experiment tag (for logging)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_data_loaders(args.data, args.batch_size, args.workers)

    if args.zerograd:
        torchvision.models.resnet.Bottleneck.forward = Bottleneck_forward

    # Model
    model = resnet50(weights=None, num_classes=1000).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # Optimizer (no scheduler)
    optimizer = build_optimizer(model.parameters(), args.opt, args.lr, args.weight_decay, tuple(args.betas))

    # TensorBoard
    # Create log directory under logs/resnet-TIMESTAMP
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"resnet50-{timestamp}-{args.tag}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)
        print(f"[eval] loss={val_loss:.4f} top1={top1:.2f} top5={top5:.2f}")
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} — lr={optimizer.param_groups[0]['lr']:.6f}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/top1", top1, epoch)
        writer.add_scalar("val/top5", top5, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  top1={top1:.2f}  top5={top5:.2f}")

        # Save checkpoint
        Path(args.save).mkdir(parents=True, exist_ok=True)
        ckpt_path = os.path.join(args.save, f"resnet50_{args.opt}_epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
