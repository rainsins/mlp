
# SimpleMLP-CPU: 纯粹、高效的 C++ 神经网络实现

**SimpleMLP-CPU** 是一个从零开始实现的感知机项目。它不依赖 PyTorch 或 TensorFlow 等重量级框架，而是直接在 CPU 上利用现代 C++ 内存模型和矩阵数学，实现了对 MNIST 手写数字集的快速训练与推理。

---

## 项目特性

* **纯粹 C++ 实现**：深度使用 C++17 标准，代码逻辑清晰，适合理解神经网络底层原理。
* **面向缓存优化**：针对 CPU L3 缓存设计的矩阵存储结构，最大化内存访问局部性。
* **灵活的训练策略**：支持 **Full-batch** 与 **Mini-batch** 自由切换，在稳定收敛与计算效率间取得平衡。
* **低依赖性**：仅依赖标准库，易于在各种 Linux/WSL2 环境中移植和编译。

---

## 技术参数

| 维度 | 规格 |
| --- | --- |
| **网络结构** | 784 (输入) -> 128 (隐藏层) -> 10 (输出) |
| **激活函数** | ReLU (隐藏层), Softmax (输出层) |
| **优化器** | 随机梯度下降 (SGD) / 全量梯度下降 |
| **数据格式** | 32-bit Float (FP32) - 针对现代 CPU SIMD 指令集优化 |
| **开发环境** | WSL2 (Ubuntu 24.04), GCC 14 / Clang 18 |

---

## 数学原理

本项目实现了标准的反向传播算法。核心计算公式如下：

### 1. 前向传播

对于每一层 $l$，其输出计算为：


$$Z^{[l]} = A^{[l-1]} \cdot W^{[l]} + b^{[l]}$$

$$A^{[l]} = \sigma(Z^{[l]})$$


其中 $\sigma$ 为激活函数（ReLU 或 Sigmoid）。

### 2. 反向传播

通过计算损失函数对权重 $W$ 的偏导数来更新参数：


$$\frac{\partial L}{\partial W} = \frac{1}{m} (A^{[l-1]})^T \cdot dZ^{[l]}$$

---

## 快速开始

### 1. 准备数据集

确保你的项目根目录下包含 MNIST 原始二进制文件：

* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`

### 2. 编译项目

使用 CMake 进行自动化构建：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

```

### 3. 运行

```bash
./mlp_trainer

```

---

## 最佳实践记录

> [!important] 为什么回归 CPU 版本？
> 1. **调试友好**：消除了异步驱动报错（如 Error 68），更易于追踪梯度消失/爆炸问题。
> 2. **延迟更低**：对于 MNIST 这种小型模型，CPU 避免了昂贵的 PCIe 显存搬运开销。
> 3. **学习率建议**：全量训练时，请务必将学习率除以样本总数（60,000），即 `lr = 0.5f / num_samples`。
> 
> 

---

## 目录结构

* `include/` - 包含 `Matrix.h`, `Network.hpp` 等核心定义。
* `src/` - 矩阵运算实现与 `main.cpp` 入口。
* `data/` - MNIST 数据集存放位置。

---

## 贡献

欢迎提交 Issue 或 Pull Request 来改进矩阵乘法的效率（例如加入 OpenMP 并行支持）！