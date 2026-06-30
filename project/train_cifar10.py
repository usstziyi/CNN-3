"""CIFAR-10 Image Classification — Complete CNN Project

This script reproduces the full Unit 7 workflow:
  1. Data loading & augmentation
  2. BaselineCNN training
  3. ResNetStyle training
  4. Training history visualization
  5. Test-set evaluation with confusion matrix
  6. Per-class accuracy breakdown
  7. Misclassification analysis
  8. Hyperparameter quick comparison
  9. Model export (TorchScript)
 10. Inference demo
 11. Model ensemble

Usage:
    python project/train_cifar10.py
"""

import copy
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

matplotlib.use("Agg")  # non-interactive backend for headless runs
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# Models
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetStyle(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 3, stride=2)
        self.layer3 = self._make_layer(128, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------
class TrainingEngine:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.best_model_state: dict | None = None

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
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
    def evaluate(
        self, loader: DataLoader, desc: str = "Evaluating",
    ) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            all_preds.append(pred.cpu())
            all_labels.append(target.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return total_loss / total, correct / total, all_preds, all_labels

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: str | None = None,
    ) -> None:
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, desc="Validating")

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if save_path:
                    self.save(save_path)
                    print(f"  >>> Best model saved (Val Acc: {val_acc:.4f})")

        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete. Best Val Acc: {self.best_val_acc:.4f}")

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.best_model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "history": self.history,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
        self.best_model_state = checkpoint["model_state_dict"]

    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, preds, labels = self.evaluate(loader, desc="Predicting")
        return preds, labels


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def build_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_size = 45000
    val_size = 5000
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {train_size:,} | Val: {val_size:,} | Test: {len(test_ds):,}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, test_loader


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_history(engines, labels, title="Training Comparison") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for engine, label in zip(engines, labels):
        h = engine.history
        axes[0].plot(h["val_loss"], label=f"{label} (best: {engine.best_val_acc:.2%})")
        axes[1].plot(h["val_acc"], label=f"{label} (best: {engine.best_val_acc:.2%})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Validation Loss"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def plot_confusion_matrix_plot(preds, labels, title="Confusion Matrix") -> None:
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                ax=ax1, cbar_kws={"label": "Count"})
    ax1.set_title(f"{title}\nConfusion Matrix (Counts)")
    ax1.set_ylabel("True Label"); ax1.set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                ax=ax2, vmin=0, vmax=1, cbar_kws={"label": "Proportion"})
    ax2.set_title(f"{title}\nConfusion Matrix (Normalized)")
    ax2.set_ylabel("True Label"); ax2.set_xlabel("Predicted Label")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")


def plot_per_class_accuracy(preds, labels, title="Per-Class Accuracy") -> None:
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    mean_acc = np.mean(per_class_acc)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if a >= mean_acc else "#e74c3c" for a in per_class_acc]
    bars = ax.bar(CIFAR10_CLASSES, per_class_acc, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=mean_acc, color="blue", linestyle="--", label=f"Mean: {mean_acc:.2%}")
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nMean Accuracy: {mean_acc:.2%}")
    ax.legend(); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, "per_class_accuracy.png")

    print("\nPer-Class Accuracy:")
    for cls, acc in zip(CIFAR10_CLASSES, per_class_acc):
        print(f"  {cls:12s}: {acc:.2%}")


def show_misclassified(preds, labels, dataset, device, max_show=10) -> None:
    incorrect_mask = preds != labels
    incorrect_indices = np.where(incorrect_mask.numpy())[0]

    if len(incorrect_indices) == 0:
        print("No errors found!")
        return

    np.random.shuffle(incorrect_indices)
    n_show = min(max_show, len(incorrect_indices))

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        idx = incorrect_indices[i]
        img, true_label = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            for c in range(3):
                img[:, :, c] = img[:, :, c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
            img = np.clip(img, 0, 1)
        pred_label = preds[idx].item()
        ax.imshow(img)
        ax.set_title(
            f"True: {CIFAR10_CLASSES[true_label]}\nPred: {CIFAR10_CLASSES[pred_label]}",
            fontsize=9, color="red",
        )
        ax.axis("off")
    fig.suptitle("Misclassified Examples", fontsize=13, color="red")
    fig.tight_layout()
    save_figure(fig, "misclassified.png")


# ---------------------------------------------------------------------------
# Hyperparameter experiment
# ---------------------------------------------------------------------------
def quick_experiment(
    model_cls, optimizer_name: str, lr: float,
    train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device, epochs: int = 10,
) -> float:
    model = model_cls().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def ensemble_predict(models, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    for m in models:
        m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Ensemble"):
            data = data.to(device)
            ensemble_logits = None
            for model in models:
                logits = model(data)
                if ensemble_logits is None:
                    ensemble_logits = F.softmax(logits, dim=1)
                else:
                    ensemble_logits += F.softmax(logits, dim=1)
            ensemble_logits /= len(models)
            preds = ensemble_logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(target)
    return torch.cat(all_preds), torch.cat(all_labels)


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
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 1. Data ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Loading CIFAR-10 dataset")
    print("=" * 60)
    train_loader, val_loader, test_loader = build_loaders()

    # ---- 2. Train BaselineCNN -----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training BaselineCNN")
    print("=" * 60)
    engine_baseline = TrainingEngine(model=BaselineCNN(), device=device)
    engine_baseline.optimizer = optim.SGD(
        engine_baseline.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4,
    )
    engine_baseline.fit(
        train_loader, val_loader, epochs=30,
        save_path=str(CHECKPOINT_DIR / "baseline_cnn.pt"),
    )

    # ---- 3. Train ResNetStyle -----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training ResNetStyle")
    print("=" * 60)
    engine_resnet = TrainingEngine(model=ResNetStyle(), device=device)
    engine_resnet.optimizer = optim.SGD(
        engine_resnet.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
    )
    engine_resnet.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        engine_resnet.optimizer, T_max=50,
    )
    engine_resnet.fit(
        train_loader, val_loader, epochs=50,
        save_path=str(CHECKPOINT_DIR / "resnet_style.pt"),
    )

    # ---- 4. Training history ------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training History")
    print("=" * 60)
    plot_training_history([engine_baseline, engine_resnet], ["BaselineCNN", "ResNetStyle"])

    # ---- 5. Test evaluation -------------------------------------------------------
    def test_report(engine, name):
        test_loss, test_acc, test_preds, test_labels = engine.evaluate(
            test_loader, desc=f"Testing {name}",
        )
        print(f"\n{'='*60}")
        print(f"  {name} - Test Results")
        print(f"{'='*60}")
        print(f"  Loss:     {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"{'='*60}")
        return test_preds, test_labels

    print("\n" + "=" * 60)
    print("  Test Set Evaluation")
    print("=" * 60)
    preds_baseline, labels_baseline = test_report(engine_baseline, "BaselineCNN")
    preds_resnet, labels_resnet = test_report(engine_resnet, "ResNetStyle")

    # ---- 6. Confusion matrix & per-class accuracy ---------------------------------
    print("\n" + "=" * 60)
    print("  Confusion Matrix (ResNetStyle)")
    print("=" * 60)
    plot_confusion_matrix_plot(preds_resnet, labels_resnet, "ResNetStyle")
    plot_per_class_accuracy(preds_resnet, labels_resnet, "ResNetStyle")

    print("\nClassification Report (ResNetStyle):")
    print(classification_report(
        labels_resnet.numpy(), preds_resnet.numpy(),
        target_names=CIFAR10_CLASSES, digits=4,
    ))

    # ---- 7. Error analysis --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Error Analysis")
    print("=" * 60)
    raw_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=None)
    show_misclassified(preds_resnet, labels_resnet, raw_test, device)

    # ---- 8. Hyperparameter quick comparison ---------------------------------------
    print("\n" + "=" * 60)
    print("  Hyperparameter Quick Comparison (10 epochs each)")
    print("=" * 60)
    configs = [
        ("SGD", 0.01), ("SGD", 0.1),
        ("Adam", 0.001), ("Adam", 0.01),
        ("AdamW", 0.001), ("AdamW", 0.01),
    ]
    results = {}
    for opt_name, lr in configs:
        acc = quick_experiment(ResNetStyle, opt_name, lr, train_loader, val_loader, device)
        results[(opt_name, lr)] = acc
        print(f"  {opt_name:6s} lr={lr:<8.4f} -> Val Acc: {acc:.4f}")

    best_config = max(results, key=results.get)
    print(f"\nBest: {best_config[0]} lr={best_config[1]} -> {results[best_config]:.4f}")

    # ---- 9. Model export ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Model Export")
    print("=" * 60)
    export_path = CHECKPOINT_DIR / "cifar10_final.pt"
    engine_resnet.save(str(export_path))

    export_model = ResNetStyle()
    checkpoint = torch.load(export_path, map_location="cpu", weights_only=False)
    export_model.load_state_dict(checkpoint["model_state_dict"])
    export_model.eval()

    scripted = torch.jit.script(export_model.cpu())
    jit_path = CHECKPOINT_DIR / "cifar10_model_scripted.pt"
    scripted.save(str(jit_path))
    print(f"TorchScript model saved to {jit_path}")
    print(f"Size: {jit_path.stat().st_size / 1e6:.2f} MB")

    # ---- 10. Inference demo -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Inference Demo")
    print("=" * 60)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                               ]))
    demo_img, demo_label = test_ds[0]
    demo_batch = demo_img.unsqueeze(0)

    with torch.no_grad():
        logits = export_model(demo_batch)
        probs = F.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()

    print(f"  True label: {CIFAR10_CLASSES[demo_label]}")
    print(f"  Predicted:  {CIFAR10_CLASSES[pred_class]}")
    print(f"  Confidence: {probs[0, pred_class].item():.2%}")
    print(f"\n  Top-5 predictions:")
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    for i in range(5):
        cls = CIFAR10_CLASSES[top5_indices[0, i].item()]
        prob = top5_probs[0, i].item()
        marker = "  <<<" if top5_indices[0, i].item() == demo_label else ""
        print(f"    {i+1}. {cls:12s}: {prob:.2%}{marker}")

    # ---- 11. Ensemble -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Model Ensemble")
    print("=" * 60)
    model1 = ResNetStyle()
    ckpt1 = torch.load(CHECKPOINT_DIR / "resnet_style.pt", map_location=device, weights_only=False)
    model1.load_state_dict(ckpt1["model_state_dict"])
    model1.to(device)

    model2 = BaselineCNN()
    ckpt2 = torch.load(CHECKPOINT_DIR / "baseline_cnn.pt", map_location=device, weights_only=False)
    model2.load_state_dict(ckpt2["model_state_dict"])
    model2.to(device)

    ensemble_preds, ensemble_labels = ensemble_predict([model1, model2], test_loader, device)
    ensemble_acc = (ensemble_preds == ensemble_labels).float().mean().item()
    print(f"Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

    print("\n" + "=" * 60)
    print("  Project complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
