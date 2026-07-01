# Pre-Activation vs Post-Activation ResNet

这两个概念主要出自 He et al. 2016 的论文 *"Identity Mappings in Deep Residual Networks"*，讨论的是**残差块（Residual Block）内部各层的排列顺序**。

---

## Post-Activation（原始 ResNet）

原始 ResNet（He et al. 2015）的残差块结构：

```
输入 x
  └→ Conv → BN → ReLU → Conv → BN ──→ (+) → ReLU → 输出
  └────────────────────────────────┘
            identity shortcut
```

**顺序**：先做卷积和 BN，**最后**再激活（ReLU 在加法之后）。因此叫 **post-activation**。

---

## Pre-Activation（改进版 ResNet）

He et al. 2016 提出的改进结构：

```
输入 x
  └→ BN → ReLU → Conv → BN → ReLU → Conv ──→ (+) → 输出
  └───────────────────────────────────────┘
            identity shortcut
```

**顺序**：**先**激活（BN → ReLU），再做卷积。因此叫 **pre-activation**——激活函数在卷积**之前**。

---

## 核心区别一览

| 特征 | Post-Activation | Pre-Activation |
|---|---|---|
| **BN + ReLU 的位置** | 放在 Conv **之后** | 放在 Conv **之前** |
| **shortcut 输出** | 经过 ReLU（非负） | 直接恒等传递（可正可负） |
| **最终激活** | 加法之后有一个 ReLU | 加法之后**没有** ReLU |
| **identity 路径** | 不是纯粹的恒等映射 | 是**纯粹的恒等映射** |
| **梯度传播** | 可能存在阻碍 | 梯度可以无阻碍地通过 shortcut |

---

## 为什么 Pre-Activation 更好？

### 1. 信息流更"干净"

Pre-Activation 中，shortcut 是**纯粹的恒等映射**——没有任何非线性变换（没有 ReLU 截断）。信息和梯度可以畅通无阻地从浅层直达深层。

```
Post-Activation shortcut:  x → ReLU(·)   # 非恒等，负值被截断
Pre-Activation  shortcut:  x → x          # 纯恒等
```

### 2. 梯度传播更顺畅

反向传播时，梯度沿 shortcut 传递不需要经过任何激活函数的导数。对于非常深的网络（1001 层等），这一差异会显著影响训练稳定性和最终性能。

### 3. 实验结果更好

论文中在 CIFAR-10 上的对比：

| 网络深度 | Post-Activation (原始) | Pre-Activation (改进) |
|---|---|---|
| 110 层 | 6.97% error | **6.37%** |
| 1001 层 | 训练困难 | **4.62%** |

---

## 直觉理解

```
Post-Activation 的问题：
  shortcut 上的 ReLU 把负值全部截为 0，
  → 残差块只能学习"加正向修正"，不能学习"减去某些东西"
  → 表达能力受限

Pre-Activation 的优势：
  shortcut 完全透明，残差分支先激活再卷积
  → 残差可以自由地加或减
  → 每一层的输入都经过 BN（更好的训练条件）
```

---

## 一句话总结

**Post-Activation**：卷积 → BN → ReLU → 加法 → **ReLU**（原始 ResNet）

**Pre-Activation**：BN → ReLU → 卷积 → 加法（无激活）（改进 ResNet）

Pre-Activation 让 shortcut 成为纯粹的恒等映射，梯度传播更畅通，在极深网络上效果明显更好。这个思想后来也深刻影响了 Transformer 中残差连接的设计。