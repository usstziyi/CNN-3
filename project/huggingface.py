from datasets import load_dataset
import torch
from torchvision import transforms

# 下载速度快得多
hf_dataset = load_dataset("uoft-cs/cifar10")

# 转换为 torchvision 兼容格式
class HFtoTorchVision(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.dataset = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]       # PIL Image
        label = item["label"]     # int
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = HFtoTorchVision(hf_dataset["train"], transform=train_transform)
test_dataset = HFtoTorchVision(hf_dataset["test"], transform=test_transform)