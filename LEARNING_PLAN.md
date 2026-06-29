# PyTorch + CNN 学习体系总览

## 项目结构

```
CNN-3/
├── pyproject.toml                          # uv 依赖配置
└── notebooks/
    ├── unit1_pytorch_basics.ipynb          # 11 markdown + 16 code cells
    ├── unit2_neural_networks.ipynb         #  9 markdown + 15 code cells
    ├── unit3_cnn_foundations.ipynb         # 13 markdown + 15 code cells
    ├── unit4_training_cnns.ipynb           # 12 markdown + 13 code cells
    ├── unit5_advanced_cnn.ipynb            # 10 markdown + 12 code cells
    ├── unit6_transfer_learning.ipynb       #  9 markdown + 12 code cells
    └── unit7_final_project.ipynb           # 13 markdown + 19 code cells
```

## 学习路径

| Unit | 主题 | 核心内容 |
|------|------|---------|
| **1** | PyTorch 基础 | Tensor 创建/运算、Autograd 自动求导、GPU 加速、梯度下降实战 |
| **2** | 神经网络基础 | nn.Module、Linear/激活/损失/优化器、MLP 训练 MNIST (97%+) |
| **3** | CNN 基础 | 卷积直观理解、Conv2d 参数详解、池化、感受野计算、SimpleCNN |
| **4** | CNN 训练实战 | Dataset/DataLoader、数据增强、Trainer 类、CIFAR-10、特征图可视化 |
| **5** | 高级 CNN 技术 | BatchNorm、Dropout、残差连接 (ResNet)、学习率调度对比、梯度裁剪 |
| **6** | 迁移学习 | 预训练模型、特征提取 vs 微调、冻结/解冻、自定义分类头 |
| **7** | 实战项目 | CIFAR-10 全流程：基线 vs ResNet → 混淆矩阵 → 超参实验 → 模型导出 → 集成 |

## 使用方法

```bash
cd CNN-3
uv sync                    # 安装所有依赖
uv run jupyter notebook    # 启动 Jupyter
```

每个 notebook 可独立运行，按 Unit 1 → 7 顺序学习效果最佳。

## 环境依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 数据集与预训练模型 |
| numpy | >=1.24.0 | 数值计算 |
| matplotlib | >=3.7.0 | 数据可视化 |
| tqdm | >=4.65.0 | 进度条 |
| scikit-learn | >=1.3.0 | 评估指标 |
| seaborn | >=0.12.0 | 高级可视化 |
| ipykernel | >=6.25.0 | Jupyter 内核 |
| jupyter | >=1.0.0 | Jupyter Notebook |

---

> 由 Trae Solo 生成
