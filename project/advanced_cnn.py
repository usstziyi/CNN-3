"""Unit 5: Advanced CNN Techniques

Covers:
  1. BatchNorm demonstration & train/eval mode
  2. Dropout regularization & p-value sweep
  3. ResidualBlock & MiniResNet
  4. Learning rate scheduler comparison
  5. OneCycleLR visualization
  6. Gradient clipping
  7. CIFAR-10 comparison: BaselineCNN vs MiniResNet

Usage:
    python project/advanced_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0
DATA_DIR = Path("./data")
FIGURE_DIR = Path("./figures")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------
# 5.1 ConvBlock with/without BatchNorm
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU()]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TestCNN(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()
        self.conv1 = ConvBlock(3, 32, use_bn)
        self.conv2 = ConvBlock(32, 64, use_bn)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 5.3 Residual Block + MiniResNet
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Quick training for comparison
# ---------------------------------------------------------------------------
def quick_train(model, train_ldr, val_ldr, device, epochs=15, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for data, target in train_ldr:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_ldr:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# 5.4 LR schedule helpers
# ---------------------------------------------------------------------------
def get_lr_schedule(scheduler_cls, lr=0.1, epochs=30, **kwargs):
    model = nn.Linear(1, 1)
    opt = optim.SGD(model.parameters(), lr=lr)
    scheduler = scheduler_cls(opt, **kwargs)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        scheduler.step()
    return lrs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # ---- 5.1 BatchNorm -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  5.1 BatchNorm: train vs eval mode")
    print("=" * 60)
    x = torch.randn(4, 16, 7, 7)
    bn = nn.BatchNorm2d(16)
    print(f"Input shape: {x.shape}")
    print(f"BN weight shape: {bn.weight.shape}, BN bias shape: {bn.bias.shape}")

    bn.train()
    out_train = bn(x)
    print(f"Train mode - mean: {out_train.mean():.4f}, std: {out_train.std():.4f}")

    bn.eval()
    with torch.no_grad():
        out_eval = bn(x)
        print(f"Eval mode  - mean: {out_eval.mean():.4f}, std: {out_eval.std():.4f}")

    # ---- BatchNorm: model comparison -----------------------------------------
    print("\n  5.1 Models with/without BatchNorm:")
    model_no_bn = TestCNN(use_bn=False)
    model_with_bn = TestCNN(use_bn=True)
    params_no = sum(p.numel() for p in model_no_bn.parameters())
    params_bn = sum(p.numel() for p in model_with_bn.parameters())
    print(f"  Without BN: {params_no:,} params")
    print(f"  With BN:    {params_bn:,} params (only +{params_bn - params_no} from BN)")

    # ---- 5.2 Dropout ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  5.2 Dropout: train vs eval behavior")
    print("=" * 60)
    x_drop = torch.ones(1, 10)
    dropout = nn.Dropout(p=0.5)

    dropout.train()
    out_train_d = dropout(x_drop)
    print(f"Train mode: {out_train_d}")
    print(f"  Non-zero elements: {(out_train_d != 0).sum().item()}")
    print(f"  Retained values ~2.0 (scaled by 1/(1-p)=2)")

    dropout.eval()
    out_eval_d = dropout(x_drop)
    print(f"Eval mode:  {out_eval_d}")
    print(f"  No dropout applied during inference")

    # Dropout p-value sweep
    p_vals = np.linspace(0, 0.9, 10)
    retained = []
    for p in p_vals:
        drop = nn.Dropout(p)
        drop.train()
        out = drop(torch.ones(10000))
        retained.append((out != 0).float().mean().item())

    fig, ax = plt.subplots()
    ax.plot(p_vals, retained, "o-")
    ax.plot([0, 0.9], [1, 0.1], "r--", label="1-p (theory)")
    ax.set_xlabel("Dropout probability (p)")
    ax.set_ylabel("Fraction of neurons retained")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "dropout_p_sweep.png")

    # ---- 5.3 Residual Block --------------------------------------------------
    print("\n" + "=" * 60)
    print("  5.3 Residual Block & MiniResNet")
    print("=" * 60)
    x_res = torch.randn(2, 16, 32, 32)
    block_same = ResidualBlock(16, 16)
    block_down = ResidualBlock(16, 32, stride=2)
    print(f"Input:            {x_res.shape}")
    print(f"Same shape block: {block_same(x_res).shape}")
    print(f"Downsample block: {block_down(x_res).shape}")

    model_res = MiniResNet()
    x_test = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_res = model_res(x_test)
    print(f"MiniResNet: {sum(p.numel() for p in model_res.parameters()):,} params")
    print(f"Input: {x_test.shape} -> Output: {out_res.shape}")

    # ---- 5.4 LR Scheduler comparison -----------------------------------------
    print("\n" + "=" * 60)
    print("  5.4 Learning Rate Schedulers")
    print("=" * 60)
    schedulers_map = {
        "StepLR (step=10, gamma=0.5)":
            get_lr_schedule(optim.lr_scheduler.StepLR, step_size=10, gamma=0.5),
        "MultiStepLR (milestones=[10,20])":
            get_lr_schedule(optim.lr_scheduler.MultiStepLR, milestones=[10, 20], gamma=0.1),
        "CosineAnnealingLR":
            get_lr_schedule(optim.lr_scheduler.CosineAnnealingLR, T_max=30),
        "CosineAnnealingWarmRestarts":
            get_lr_schedule(optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=10, T_mult=2),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, lrs) in zip(axes.flat, schedulers_map.items()):
        ax.plot(lrs, "o-", markersize=3)
        ax.set_title(name)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Learning Rate Schedulers Comparison", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "lr_schedulers_comparison.png")

    # OneCycleLR
    print("\n  OneCycleLR:")
    mdl = nn.Linear(1, 1)
    opt = optim.SGD(mdl.parameters(), lr=0.01)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=30)
    lrs_onecycle = []
    for _ in range(30):
        lrs_onecycle.append(opt.param_groups[0]["lr"])
        sched.step()

    fig, ax = plt.subplots()
    ax.plot(lrs_onecycle, "o-", markersize=3)
    ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("OneCycleLR (warm-up + decay)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "one_cycle_lr.png")

    # ---- 5.5 Gradient Clipping -----------------------------------------------
    print("\n" + "=" * 60)
    print("  5.5 Gradient Clipping")
    print("=" * 60)
    model_clip = MiniResNet().to(device)
    x_clip = torch.randn(2, 3, 32, 32).to(device)
    y_clip = torch.randint(0, 10, (2,)).to(device)

    criterion = nn.CrossEntropyLoss()
    opt_clip = optim.Adam(model_clip.parameters(), lr=0.01)

    output = model_clip(x_clip)
    loss = criterion(output, y_clip)
    opt_clip.zero_grad()
    loss.backward()

    total_norm_before = 0
    for p in model_clip.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before ** 0.5

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm)

    total_norm_after = 0
    for p in model_clip.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after ** 0.5

    print(f"Gradient norm before clipping: {total_norm_before:.4f}")
    print(f"Gradient norm after clipping:  {total_norm_after:.4f} (max={max_norm})")
    opt_clip.step()

    # ---- 5.6 CIFAR-10 comparison ---------------------------------------------
    print("\n" + "=" * 60)
    print("  5.6 CIFAR-10: BaselineCNN vs MiniResNet")
    print("=" * 60)

    train_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform_aug)
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_size = 45000
    val_size = 5000
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(full_test, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print("Training models for comparison...")
    print("-" * 50)

    acc_baseline = quick_train(BaselineCNN(), train_loader, val_loader, device)
    print(f"Baseline CNN:                     Val Acc = {acc_baseline:.2%}")

    acc_resnet = quick_train(MiniResNet(), train_loader, val_loader, device)
    print(f"MiniResNet (BN + Residual):        Val Acc = {acc_resnet:.2%}")

    print("-" * 50)
    print(f"Improvement: +{(acc_resnet - acc_baseline)*100:.1f}% absolute")

    print("\n" + "=" * 60)
    print("  Unit 5 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
