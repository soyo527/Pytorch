import torch
from torch import nn


class LeNet(nn.Module):
    """
    一个适用于 FashionMNIST 的 LeNet 风格卷积神经网络。

    输入图片大小是 1 x 28 x 28，输出长度为 10，
    分别对应 FashionMNIST 的 10 个类别。
    """

    def __init__(self):
        # 调用父类 nn.Module 的初始化函数。
        super().__init__()

        # 第 1 个卷积层：
        # 输入通道数 = 1（灰度图只有 1 个通道）
        # 输出通道数 = 6（会学出 6 组特征）
        # 卷积核大小 = 5 x 5
        # padding = 2 的作用是让特征图的高宽保持 28 x 28 不变
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # LeNet 经典结构里常用 Sigmoid 作为激活函数。
        # 激活函数的作用是给网络增加非线性表达能力。
        self.sig = nn.Sigmoid()

        # 第 1 个平均池化层：
        # 把特征图从 28 x 28 缩小到 14 x 14，减少计算量。
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第 2 个卷积层：
        # 输入通道数 = 6，输出通道数 = 16，卷积核大小 = 5 x 5。
        # 这里没有 padding，所以高宽会从 14 x 14 变成 10 x 10。
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 第 2 个平均池化层：
        # 把 10 x 10 再缩小到 5 x 5。
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 把卷积层输出的 3 维特征图拉平成 1 维向量，
        # 这样才能接到后面的全连接层。
        self.flatten = nn.Flatten()

        # 进入全连接层时，特征图大小是 16 x 5 x 5，
        # 所以输入特征数 = 16 * 5 * 5 = 400。
        self.f5 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)

        # 最后一层输出 10 个数。
        # 这 10 个数叫 logits，表示模型对 10 个类别的“打分”。
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        定义数据在网络中“前向传播”的路线。

        参数:
            x: 输入图片张量，形状是 [batch_size, 1, 28, 28]

        返回:
            输出张量，形状是 [batch_size, 10]
        """

        # 卷积 + 激活之后：
        # [batch_size, 1, 28, 28] -> [batch_size, 6, 28, 28]
        x = self.sig(self.c1(x))

        # 池化之后：
        # [batch_size, 6, 28, 28] -> [batch_size, 6, 14, 14]
        x = self.s2(x)

        # 第 2 次卷积 + 激活：
        # [batch_size, 6, 14, 14] -> [batch_size, 16, 10, 10]
        x = self.sig(self.c3(x))

        # 第 2 次池化：
        # [batch_size, 16, 10, 10] -> [batch_size, 16, 5, 5]
        x = self.s4(x)

        # 拉平成一维向量：
        # [batch_size, 16, 5, 5] -> [batch_size, 400]
        x = self.flatten(x)

        # 通过 3 个全连接层，逐步把特征映射到 10 个类别分数。
        x = self.f5(x)  # [batch_size, 400] -> [batch_size, 120]
        x = self.f6(x)  # [batch_size, 120] -> [batch_size, 84]
        x = self.f7(x)  # [batch_size, 84] -> [batch_size, 10]

        return x


if __name__ == "__main__":
    # 先判断当前机器能不能用 GPU，能用就用 GPU，否则退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并放到对应设备上。
    model = LeNet().to(device)

    # torchsummary 只是“查看网络结构”的辅助包，不是模型运行本身必须的依赖。
    # 所以把它放到这里按需导入，避免用户在没装 torchsummary 时无法导入 LeNet。
    try:
        from torchsummary import summary
    except ImportError:
        print("当前环境未安装 torchsummary，暂时无法打印网络结构摘要。")
    else:
        # 打印网络结构摘要，方便快速查看每一层的输入输出形状。
        print(summary(model, input_size=(1, 28, 28)))
