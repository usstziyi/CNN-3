"""Unit 4: CNN Training in Practice

Covers:
  1. Custom Dataset (SyntheticImageDataset)
  2. DataLoader parameters
  3. Transforms & data augmentation
  4. Train/val split
  5. CIFAR10_CNN model
  6. Trainer class with save/load
  7. Training history visualization
  8. Test-set evaluation
  9. Feature map visualization

Usage:
    python project/train_cnns.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11

SEED = 42
BATCH_SIZE = 128
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


# ---------------------------------------------------------------------------
# 4.2 Custom Dataset
# ---------------------------------------------------------------------------
class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 4.5 CIFAR-10 CNN Model
# ---------------------------------------------------------------------------
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 4.6 Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc="Training", leave=False)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader, desc="Evaluating"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs, save_path=None):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader, desc="Validating")

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {lr:.6f}")

            if val_acc > self.best_val_acc and save_path:
                self.best_val_acc = val_acc
                self.save(save_path)
                print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint from {path} (best_val_acc={self.best_val_acc:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def denormalize(img_tensor, mean, std):
    img = img_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.clamp(0, 1)


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------
# 4.9 Feature map visualization
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_feature_maps(model, layer_name, input_tensor):
    model.eval()
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    hook_handles = []
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handles.append(module.register_forward_hook(hook_fn(name)))

    _ = model(input_tensor)

    for h in hook_handles:
        h.remove()

    return features.get(layer_name, None)


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

    # ---- 4.2 Custom Dataset demo ---------------------------------------------
    print("\n" + "=" * 60)
    print("  4.2 Custom Dataset Demo")
    print("=" * 60)
    dataset = SyntheticImageDataset(num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Sample shape: {img.shape}, label: {label}")

    loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    # 将 DataLoader 转换为迭代器
    # 从迭代器中取出下一个元素（即一个 batch）
    batch = next(iter(loader))
    data, labels = batch
    print(f"Batch data: {data.shape}")
    print(f"Batch labels: {labels.shape}")
    print(f"Number of batches: {len(loader)}")

    # ---- 4.3 Transforms & data loading ---------------------------------------
    print("\n" + "=" * 60)
    print("  4.3 Data Loading with Transforms")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset_full = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    print(f"Train: {len(train_dataset_full):,}, Test: {len(test_dataset):,}")
    print(f"Classes: {train_dataset_full.classes}")

    # ---- Data augmentation visualization -------------------------------------
    print("\n" + "=" * 60)
    print("  4.3 Data Augmentation Visualization")
    print("=" * 60)
    demo_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    demo_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    original_img = demo_dataset[0][0]

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(1, 5):
        augmented = demo_aug(original_img)
        axes[i].imshow(augmented.permute(1, 2, 0))
        axes[i].set_title(f"Augmented #{i}")
        axes[i].axis("off")
    fig.suptitle("Data Augmentation Examples")
    fig.tight_layout()
    save_figure(fig, "augmentation_demo.png")

    # ---- CIFAR-10 samples grid -----------------------------------------------
    raw_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        img_raw, label_raw = raw_dataset[i]
        ax = axes[i // 5][i % 5]
        ax.imshow(img_raw.permute(1, 2, 0))
        ax.set_title(f"{CIFAR10_CLASSES[label_raw]}")
        ax.axis("off")
    fig.suptitle("CIFAR-10 Samples")
    fig.tight_layout()
    save_figure(fig, "cifar10_samples.png")

    # ---- 4.4 Train/val split ------------------------------------------------
    print("\n" + "=" * 60)
    print("  4.4 Train / Validation Split")
    print("=" * 60)
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # ---- 4.5 Model -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  4.5 CIFAR10_CNN Model")
    print("=" * 60)
    model = CIFAR10_CNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input {x.shape} -> Output {out.shape}")

    # ---- 4.6 Training --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  4.6 Training (20 epochs)")
    print("=" * 60)
    model = CIFAR10_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    trainer = Trainer(model, device, criterion, optimizer, scheduler)
    trainer.fit(train_loader, val_loader, epochs=20,
                save_path=str(CHECKPOINT_DIR / "cifar10_cnn.pt"))

    # ---- 4.7 Training curves -------------------------------------------------
    print("\n" + "=" * 60)
    print("  4.7 Training History")
    print("=" * 60)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(trainer.history["train_loss"], "b-", label="Train")
    axes[0].plot(trainer.history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(trainer.history["train_acc"], "b-", label="Train")
    axes[1].plot(trainer.history["val_acc"], "r-", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"CIFAR10 CNN Training (Best Val Acc: {trainer.best_val_acc:.2%})", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "training_history_cifar10_cnn.png")

    # ---- 4.8 Test evaluation -------------------------------------------------
    print("\n" + "=" * 60)
    print("  4.8 Test Set Evaluation")
    print("=" * 60)
    ckpt_path = CHECKPOINT_DIR / "cifar10_cnn.pt"
    if ckpt_path.exists():
        model_eval = CIFAR10_CNN().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_eval.load_state_dict(ckpt["model_state_dict"])
        trainer.model = model_eval

    test_loss, test_acc = trainer.evaluate(test_loader, desc="Testing")
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Test Acc:   {test_acc:.4f} ({test_acc*100:.2f}%)")

    # ---- 4.9 Feature map visualization ---------------------------------------
    print("\n" + "=" * 60)
    print("  4.9 Feature Map Visualization")
    print("=" * 60)
    sample_img, sample_label = test_dataset[0]
    sample_batch = sample_img.unsqueeze(0).to(device)

    fmaps = visualize_feature_maps(model_eval, "conv_block1.0", sample_batch)
    if fmaps is not None:
        fmaps = fmaps.cpu().squeeze(0)
        n = min(16, fmaps.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.imshow(fmaps[i], cmap="viridis")
            ax.axis("off")
        fig.suptitle(f"Feature Maps: conv_block1.0 ({n} of {fmaps.shape[0]} channels)")
        fig.tight_layout()
        save_figure(fig, "feature_maps_conv1.png")

    print("\n" + "=" * 60)
    print("  Unit 4 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
