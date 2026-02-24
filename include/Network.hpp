#include "Matrix.h"
#include <vector>
#include <iostream>

// sycl::queue q(sycl::gpu_selector_v); 

class Network
{
public:
    Matrix W1, b1; // 输入层到隐藏层 (128x784, 128x1)
    Matrix W2, b2; // 隐藏层到输出层 (10x128, 10x1)
    float learning_rate;
    Matrix weights1; 
    Matrix weights2;
    Matrix bias1;
    Matrix bias2;

    Network(float lr) : W1(128, 784, true), b1(128, 1),
                         W2(10, 128, true), b2(10, 1),
                         weights1(784, 128), weights2(128, 10), 
                        bias1(1, 128), bias2(1, 10),
                         learning_rate(lr) {}

    // 训练主循环
    void train(const std::vector<Matrix> &images, const std::vector<int> &labels, int epochs)
    {
        for (int e = 0; e < epochs; ++e)
        {
            float total_loss = 0;
            int correct = 0;

            for (size_t i = 0; i < images.size(); ++i)
            {
                // --- 1. 前向传播 ---
                // Hidden Layer: z1 = W1*x + b1, a1 = ReLU(z1)
                Matrix z1 = Matrix::multiply(W1, images[i]);
                for (int j = 0; j < 128; ++j)
                    z1.data[j] += b1.data[j];

                Matrix a1(128, 1);
                for (int j = 0; j < 128; ++j)
                    a1.data[j] = std::max(0.0f, z1.data[j]);

                // Output Layer: z2 = W2*a1 + b2, a2 = Softmax(z2)
                Matrix z2 = Matrix::multiply(W2, a1);
                for (int j = 0; j < 10; ++j)
                    z2.data[j] += b2.data[j];

                std::vector<float> a2 = Activations::softmax(z2.data);

                // 计算 Loss (Cross Entropy) 和 准确率
                total_loss -= std::log(a2[labels[i]] + 1e-15);
                if (std::distance(a2.begin(), std::max_element(a2.begin(), a2.end())) == labels[i])
                {
                    correct++;
                }

                // --- 2. 反向传播 ---
                // 输出层梯度: dZ2 = a2 - y (y 是 one-hot 向量)
                Matrix dZ2(10, 1);
                for (int j = 0; j < 10; ++j)
                    dZ2.data[j] = a2[j];
                dZ2.data[labels[i]] -= 1.0;

                // 隐藏层梯度: dZ1 = (W2^T * dZ2) * ReLU'(z1)
                Matrix W2T = W2.transpose();
                Matrix dZ1_raw = Matrix::multiply(W2T, dZ2);
                Matrix dZ1(128, 1);
                for (int j = 0; j < 128; ++j)
                {
                    dZ1.data[j] = (z1.data[j] > 0) ? dZ1_raw.data[j] : 0;
                }

                // --- 3. 更新参数 (梯度下降) ---
                updateParameters(images[i], a1, dZ1, dZ2);
            }
            std::cout << "Epoch " << e << " | Loss: " << total_loss / images.size()
                      << " | Accuracy: " << (float)correct / images.size() * 100 << "%" << std::endl;
        }
    }

    // 在 Network 类中添加
    std::vector<float> predict(const Matrix &x)
    {
        // 1. 隐藏层计算 (ReLU 激活)
        Matrix z1 = Matrix::multiply(W1, x);
        for (int j = 0; j < 128; ++j)
            z1.data[j] += b1.data[j];

        Matrix a1(128, 1);
        for (int j = 0; j < 128; ++j)
            a1.data[j] = std::max(0.0f, z1.data[j]);

        // 2. 输出层计算 (Softmax 激活)
        Matrix z2 = Matrix::multiply(W2, a1);
        for (int j = 0; j < 10; ++j)
            z2.data[j] += b2.data[j];

        // 返回 Softmax 后的概率分布
        return Activations::softmax(z2.data);
    }

private:
    void updateParameters(const Matrix &x, const Matrix &a1, const Matrix &dZ1, const Matrix &dZ2)
    {
        // W2 = W2 - lr * (dZ2 * a1^T)
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 128; ++j)
            {
                W2(i, j) -= learning_rate * dZ2.data[i] * a1.data[j];
            }
            b2.data[i] -= learning_rate * dZ2.data[i];
        }

        // W1 = W1 - lr * (dZ1 * x^T)
        for (int i = 0; i < 128; ++i)
        {
            for (int j = 0; j < 784; ++j)
            {
                W1(i, j) -= learning_rate * dZ1.data[i] * x.data[j];
            }
            b1.data[i] -= learning_rate * dZ1.data[i];
        }
    }
};