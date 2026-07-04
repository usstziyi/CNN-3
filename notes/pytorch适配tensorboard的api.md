根据 PyTorch 官方文档（`torch.utils.tensorboard.SummaryWriter`），PyTorch 适配 TensorBoard 一共提供了 **14 个** `add_` 系列的 API。

以下是完整列表：

| # | API 方法 | 功能说明 |
|---|---------|---------|
| 1 | `add_scalar` | 添加单个标量数据（如 loss、accuracy） |
| 2 | `add_scalars` | 添加多个标量数据到同一图表 |
| 3 | `add_histogram` | 添加直方图（如权重/梯度分布） |
| 4 | `add_image` | 添加单张图像 |
| 5 | `add_images` | 添加一批图像（你示例中使用的方法） |
| 6 | `add_figure` | 添加 matplotlib figure |
| 7 | `add_video` | 添加视频数据 |
| 8 | `add_audio` | 添加音频数据 |
| 9 | `add_text` | 添加文本字符串 |
| 10 | `add_graph` | 添加模型计算图 |
| 11 | `add_embedding` | 添加嵌入向量（用于降维可视化） |
| 12 | `add_pr_curve` | 添加 Precision-Recall 曲线 |
| 13 | `add_mesh` | 添加 3D 网格/点云数据 |
| 14 | `add_hparams` | 添加超参数及其对应指标 |

> **注意**：除了这 14 个 `add_*` 方法外，`SummaryWriter` 还有一些非 `add_` 前缀的辅助方法，如 `flush()`、`close()`、`get_logdir()` 等，但它们不属于数据记录 API。

所以你提到的 `writer.add_images("CIFAR-10/Samples", grid, global_step=3)` 正是这 14 个 API 中的第 5 个。