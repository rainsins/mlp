#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
// #include <sycl/sycl.hpp>

class Matrix
{
public:
    int rows, cols;
    std::vector<float> data;

    Matrix(int r, int c, bool init_rand = false) : rows(r), cols(c), data(r * c, 0.0f)
    {
        if (init_rand)
        {
            // He Initialization: 适合 ReLU 激活函数
            std::default_random_engine gen(std::random_device{}());
            std::normal_distribution<float> dist(0.0f, std::sqrt(2.0 / r));
            for (auto &val : data)
                val = dist(gen);
        }
    }

    // 添加这个：允许 A(i, j) 访问并修改数据
    float &operator()(int r, int c)
    {
        return data[r * cols + c];
    }

    // 添加这个：允许 const 对象访问数据
    const float &operator()(int r, int c) const
    {
        return data[r * cols + c];
    }
    
    // 矩阵乘法：C = A * B
    // 采用 i-k-j 顺序，极大提高 CPU 缓存命中率
    static Matrix multiply(const Matrix &A, const Matrix &B)
    {
        Matrix C(A.rows, B.cols);
        for (int i = 0; i < A.rows; ++i)
        {
            for (int k = 0; k < A.cols; ++k)
            {
                float temp = A.data[i * A.cols + k];
                for (int j = 0; j < B.cols; ++j)
                {
                    C.data[i * B.cols + j] += temp * B.data[k * B.cols + j];
                }
            }
        }
        return C;
    }

    // 在 Matrix 类定义中添加
    Matrix transpose() const
    {
        // 创建一个行列互换的新矩阵
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                // 原矩阵的 (i, j) 变成新矩阵的 (j, i)
                result.data[j * rows + i] = data[i * cols + j];
            }
        }
        return result;
    }
};

class Activations
{
public:
    static std::vector<float> softmax(const std::vector<float> &z)
    {
        std::vector<float> a(z.size());

        // 1. 找到最大值 M
        float max_val = *std::max_element(z.begin(), z.end());

        // 2. 计算 exp(z_i - M) 并求和
        float sum = 0.0f;
        for (size_t i = 0; i < z.size(); ++i)
        {
            a[i] = std::exp(z[i] - max_val);
            sum += a[i];
        }

        // 3. 归一化
        for (size_t i = 0; i < z.size(); ++i)
        {
            a[i] /= (sum + 1e-15); // 添加极小值防止除零
        }

        return a;
    }
};

#endif