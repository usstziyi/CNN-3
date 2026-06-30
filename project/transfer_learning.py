"""Unit 6: Transfer Learning & Fine-tuning

Covers:
  1. Check available pretrained models
  2. Feature extraction: freeze backbone, train FC only
  3. Fine-tuning: unfreeze layer4 + fc with smaller LR
  4. Custom classifier head design
  5. Freeze/unfreeze utility functions
  6. Test-set evaluation

Usage:
    python project/transfer_learning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11

SEED = 42
NUM_WORKERS = 0
DATA_DIR = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
FIGURE_DIR = Path("./figures")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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
# 6.7 Custom Classifier Head
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# 6.8 Freeze / Unfreeze utilities
# ---------------------------------------------------------------------------
def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_all(model):
    set_requires_grad(model, False)


def unfreeze_all(model):
    set_requires_grad(model, True)


def freeze_until(model, layer_name):
    freeze = True
    for name, param in model.named_parameters():
        if layer_name in name:
            freeze = False
        param.requires_grad = not freeze


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(loader, desc="Eval", leave=False):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 6.3 Check available pretrained models -------------------------------
    print("\n" + "=" * 60)
    print("  6.3 Available Pretrained Models")
    print("=" * 60)
    available_weights = {
        "resnet18": models.ResNet18_Weights,
        "resnet50": models.ResNet50_Weights,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights,
        "efficientnet_b0": models.EfficientNet_B0_Weights,
    }
    for name, weights_cls in available_weights.items():
        try:
            weights = weights_cls.DEFAULT
            print(f"  {name:25s}: available")
        except Exception as e:
            print(f"  {name:25s}: NOT available ({e})")

    # ---- 6.4 Feature Extraction ----------------------------------------------
    print("\n" + "=" * 60)
    print("  6.4 Feature Extraction with ResNet-18")
    print("=" * 60)

    weights = models.ResNet18_Weights.DEFAULT
    pretrained_transform = weights.transforms()
    print(f"Pretrained preprocessing: {pretrained_transform}")

    model_fe = models.resnet18(weights=weights)
    print(f"Original last layer: {model_fe.fc}")

    num_features = model_fe.fc.in_features
    model_fe.fc = nn.Linear(num_features, 10)
    print(f"New last layer: {model_fe.fc}")

    freeze_all(model_fe)
    for param in model_fe.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_fe.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)")

    # ---- Data loading --------------------------------------------------------
    model_fe = model_fe.to(device)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_size = 45000
    val_size = 5000
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {train_size:,}, Val: {val_size:,}, Test: {len(test_dataset):,}")

    # ---- Train feature extraction --------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)

    print("\nFeature extraction mode (freeze backbone, train FC only)...")
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model_fe, train_loader, criterion, optimizer_fe, device)
        val_loss, val_acc = evaluate(model_fe, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ---- 6.5 Fine-tuning -----------------------------------------------------
    print("\n" + "=" * 60)
    print("  6.5 Fine-tuning (unfreeze layer4 + fc)")
    print("=" * 60)

    model_ft = models.resnet18(weights=weights)
    num_features_ft = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features_ft, 10)
    model_ft = model_ft.to(device)

    print("layer4 + fc trainable, rest frozen:")
    for name, param in model_ft.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
            print(f"  Trainable: {name}")
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_ft.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)")

    params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(params_to_update, lr=0.0001)

    print("\nFine-tuning mode (layer4 + fc)...")
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model_ft, train_loader, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate(model_ft, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ---- 6.6 Test evaluation -------------------------------------------------
    print("\n" + "=" * 60)
    print("  6.6 Test Set Evaluation (Fine-tuned ResNet-18)")
    print("=" * 60)
    test_loss, test_acc = evaluate(model_ft, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    print(f"\nComparison:")
    print(f"  Unit 4 self-trained CNN:  ~75-80%")
    print(f"  Unit 5 MiniResNet:        ~85-88%")
    print(f"  Transfer learning ResNet-18: ~90-93%")

    # ---- 6.7 Custom classifier head ------------------------------------------
    print("\n" + "=" * 60)
    print("  6.7 Custom Classifier Head Design")
    print("=" * 60)
    model_custom_head = models.resnet18(weights=weights)
    num_features_custom = model_custom_head.fc.in_features
    model_custom_head.fc = ClassifierHead(num_features_custom, 10)
    print(f"Custom classification head:\n{model_custom_head.fc}")

    # ---- 6.8 Freeze/Unfreeze utilities ---------------------------------------
    print("\n" + "=" * 60)
    print("  6.8 Freeze / Unfreeze Utility Functions")
    print("=" * 60)
    print("  freeze_all(model)           - freeze all parameters")
    print("  unfreeze_all(model)         - unfreeze all parameters")
    print("  freeze_until(model, name)  - freeze layers before 'name'")

    demo_model = models.resnet18(weights=None)
    freeze_all(demo_model)
    print(f"  After freeze_all: all require_grad = {all(not p.requires_grad for p in demo_model.parameters())}")
    unfreeze_all(demo_model)
    print(f"  After unfreeze_all: all require_grad = {all(p.requires_grad for p in demo_model.parameters())}")
    freeze_until(demo_model, "layer4")
    frozen_count = sum(1 for p in demo_model.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in demo_model.parameters() if p.requires_grad)
    print(f"  After freeze_until('layer4'): {frozen_count} frozen, {trainable_count} trainable")

    # ---- Save checkpoint -----------------------------------------------------
    save_path = CHECKPOINT_DIR / "resnet18_finetuned.pt"
    torch.save({"model_state_dict": model_ft.state_dict(), "val_acc": val_acc}, save_path)
    print(f"\nFine-tuned model saved to {save_path}")

    print("\n" + "=" * 60)
    print("  Unit 6 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
