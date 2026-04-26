from pathlib import Path

import numpy as np
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 定位当前脚本所在的项目目录
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def preview_fashion_mnist():
    """
    随机取出一批 FashionMNIST 训练图片，并把它们可视化出来。

    这个脚本不参与训练，它只是帮助你直观看到：
    数据集里的图片长什么样、类别标签对应什么衣物。
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("运行 plot.py 需要 matplotlib，请先安装 matplotlib。") from exc

    train_data = FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        transform=transforms.Compose(
            [
                # 这里把图片放大到 224x224，只是为了展示更清楚。
                # 注意：这不代表模型训练时必须用 224x224。
                transforms.Resize(size=224),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    # 只取一个 batch 来展示即可，不需要遍历完整个数据集。
    for batch_x, batch_y in train_loader:
        break

    # squeeze() 用来去掉“通道数为 1”这一维，
    # 方便后面用 matplotlib 直接显示灰度图。
    batch_x = batch_x.squeeze().numpy()
    batch_y = batch_y.numpy()

    # classes 是 FashionMNIST 自带的类别名称列表。
    class_label = train_data.classes

    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=10)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()


if __name__ == "__main__":
    preview_fashion_mnist()
