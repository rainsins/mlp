#include "MNISTReader.hpp"
#include "Network.hpp"

void verify_prediction(Network &net, const Matrix &img, int true_label)
{
    // 1. 在终端用字符画出这张图 (28x28)
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            std::cout << (img.data[i * 28 + j] > 0.5 ? "##" : "  ");
        }
        std::cout << std::endl;
    }

    // 2. 前向传播获取预测结果
    // 假设你已经封装好了 forward 函数返回 std::vector<float> (softmax 结果)
    auto probs = net.predict(img);
    int prediction = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

    std::cout << "真实标签: " << true_label << std::endl;
    std::cout << "模型预测: " << prediction << std::endl;
}

int main()
{
    MNISTReader reader;
    auto train_images = reader.readImages("/home/rainsin/Cpp/data/train-images-idx3-ubyte");
    auto train_labels = reader.readLabels("/home/rainsin/Cpp/data/train-labels-idx1-ubyte");

    Network net(0.01f);                         // 学习率 0.01
    net.train(train_images, train_labels, 30); // 训练 10 轮

    auto test_images = reader.readImages("/home/rainsin/Cpp/data/t10k-images-idx3-ubyte");
    auto test_labels = reader.readLabels("/home/rainsin/Cpp/data/t10k-labels-idx1-ubyte");

    int test_correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i)
    {
        // 只需要前向传播
        // 假设你在 Network 类里写了 predict 函数，或者直接复用 forward 逻辑
        auto probs = net.predict(test_images[i]);
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        if (pred == test_labels[i])
            test_correct++;
    }

    std::cout << "------------------------------------" << std::endl;
    std::cout << "期末考试（测试集）准确率: " << (float)test_correct / test_images.size() * 100 << "%" << std::endl;

    std::cout << "\n--- 抽查测试集中的第 100 张图片 ---" << std::endl;
    // 调用函数：传入网络实例、图片矩阵、真实标签
    verify_prediction(net, test_images[99], test_labels[99]);

    std::cout << "\n--- 抽查测试集中的第 500 张图片 ---" << std::endl;
    verify_prediction(net, test_images[499], test_labels[499]);
    return 0;
}