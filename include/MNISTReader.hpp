#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "Matrix.h"

class MNISTReader {
private:
    // 大端序转小端序 (MNIST 文件头是 32 位整数)
    uint32_t swapEndian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
        return (val << 16) | (val >> 16);
    }

public:
    // 读取图像文件
    // 返回一个 vector，每个元素是一个 784x1 的 Matrix
    std::vector<Matrix> readImages(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open images file");

        uint32_t magic, num_images, rows, cols;
        file.read((char*)&magic, 4);
        file.read((char*)&num_images, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);

        num_images = swapEndian(num_images);
        rows = swapEndian(rows);
        cols = swapEndian(cols);

        int image_size = rows * cols; // 784
        std::vector<Matrix> images;
        
        for (int i = 0; i < num_images; ++i) {
            Matrix img(image_size, 1);
            for (int j = 0; j < image_size; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                img.data[j] = pixel / 255.0; // 归一化
            }
            images.push_back(img);
        }
        return images;
    }

    // 读取标签文件
    std::vector<int> readLabels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open labels file");

        uint32_t magic, num_labels;
        file.read((char*)&magic, 4);
        file.read((char*)&num_labels, 4);
        num_labels = swapEndian(num_labels);

        std::vector<int> labels(num_labels);
        for (int i = 0; i < num_labels; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, 1);
            labels[i] = (int)label;
        }
        return labels;
    }
};

#endif