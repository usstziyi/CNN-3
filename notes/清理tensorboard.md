在 Jupyter Notebook 中终止 TensorBoard 有以下几种方法，按推荐程度排序：

### 1. 直接在单元格中执行 kill 命令（最快）

提示语已经告诉你 PID 了，直接运行：

```bash
!kill 55457
```

或者使用 `%tensorboard` 自带的停止方式：

```python
%tensorboard --logdir runs --port 6006 --bind_all  # 重新执行时加 --force-reload
```

### 2. 通过 tensorboard 管理器停止（最规范）

```python
from tensorboard import notebook
notebook.stop(port=6006)      # 停止指定端口
# notebook.stop(pid=55457)    # 或按 PID 停止
# notebook.list()             # 查看所有正在运行的实例
```

### 3. 批量清理所有残留进程

如果多次启动导致端口占用混乱：

```bash
# 杀掉所有 tensorboard 进程
!pkill -f tensorboard

# 确认是否清理干净
!ps aux | grep tensorboard
```

### ⚠️ 注意事项

-   **不要只关闭浏览器标签页**：那只是断开前端显示，后端进程仍在运行并占用端口。
-   **重启 Kernel 不会自动杀死 TensorBoard**：TensorBoard 是独立于 Jupyter Kernel 的系统进程，Kernel 重启后它依然存活。这也是为什么你会看到 `Reusing TensorBoard on port 6006 (pid 55457)` 这条提示。
-   **端口被占但找不到进程**：用 `!lsof -i :6006` 查看实际占用端口的 PID，再精准 kill。

> **💡 最佳实践**：每次实验结束或切换 logdir 前，养成先用 `notebook.stop()` 或 `!kill` 清理旧实例的习惯，避免端口冲突和内存泄漏。