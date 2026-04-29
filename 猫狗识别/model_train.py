import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder


from model import GoogLeNet,Inception



# 项目根目录。这样写的好处是：无论你把项目拷到哪里，代码都能根据当前文件所在位置自动找到数据和模型文件。
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "train"
MODEL_SAVE_PATH = PROJECT_ROOT / "best_model.pth"


def train_val_data_process():
    # 读取官方训练集（共 60000 张图片）。
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.162, 0.151, 0.138],
            std=[0.058, 0.052, 0.048]
        ),
    ])

    full_train_data=ImageFolder(DATA_ROOT,transform=train_transform)

    # 按 8:2 划分训练集和验证集。
    train_size = round(0.8 * len(full_train_data))
    val_size = len(full_train_data) - train_size

    # random_split 会返回两个子数据集。
    train_subset, val_subset = Data.random_split(full_train_data, [train_size, val_size])

    # 训练集 DataLoader：
    # shuffle=True 表示每个 epoch 都把样本顺序打乱，
    # 这样模型不容易记住固定顺序。
    train_dataloader = Data.DataLoader(
        dataset=train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    # 验证集 DataLoader：
    # 验证时不需要打乱顺序，所以 shuffle=False。
    val_dataloader = Data.DataLoader(
        dataset=val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    执行完整的模型训练流程，并返回每个 epoch 的指标记录。

    参数:
        model: 要训练的神经网络
        train_dataloader: 训练集加载器
        val_dataloader: 验证集加载器
        num_epochs: 训练轮数
    """

    # 选择训练设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adam 是一种常用优化器，负责根据梯度更新参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 交叉熵损失适合做多分类任务。
    # 注意：CrossEntropyLoss 期望输入的是 logits，
    # 所以模型最后一层不需要再手动做 Softmax。
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # 先保存一份“当前最优模型参数”。
    # 训练过程中只要验证集准确率刷新了，就更新这份参数。
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 下面这些列表用来记录每个 epoch 的损失和准确率，方便后面画图。
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    # since 用来统计训练总耗时。
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 每个 epoch 开始前都把累计量清零。
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # -------------------- 训练阶段 --------------------
        # train() 会把模型切换到训练模式。
        # 像 BatchNorm、Dropout 这类层在训练和测试时行为不同，
        # 所以这是一个很重要的习惯。
        model.train()

        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播：输入图片，得到 10 个类别分数。
            output = model(batch_x)

            # argmax(dim=1) 表示在“类别维度”上找最大值的下标，
            # 也就是模型预测的类别编号。
            pre_lab = torch.argmax(output, dim=1)

            # 计算损失：比较“模型输出”和“真实标签”之间的差距。
            loss = criterion(output, batch_y)

            # 反向传播前先把上一轮残留的梯度清零。
            optimizer.zero_grad()

            # 反向传播：根据损失计算每个参数的梯度。
            loss.backward()

            # 用优化器根据梯度更新参数。
            optimizer.step()

            # 下面几行是在“累计这个 epoch 的总损失和总正确数”。
            train_loss += loss.item() * batch_y.size(0)
            train_corrects += torch.sum(pre_lab == batch_y)
            train_num += batch_x.size(0)

        # -------------------- 验证阶段 --------------------
        # eval() 会把模型切换到评估模式。
        model.eval()

        # 验证阶段不需要计算梯度，用 no_grad() 可以节省显存和计算。
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, batch_y)

                val_loss += loss.item() * batch_y.size(0)
                val_corrects += torch.sum(pre_lab == batch_y)
                val_num += batch_x.size(0)

        # 用“总损失 / 样本总数”得到平均损失。
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        # 用“预测正确的数量 / 总样本数”得到准确率。
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(
            f"Train Loss: {train_loss_all[-1]:.4f} | "
            f"Train Acc: {train_acc_all[-1]:.4f}"
        )
        print(
            f"Val Loss:   {val_loss_all[-1]:.4f} | "
            f"Val Acc:   {val_acc_all[-1]:.4f}"
        )

        # 如果这一轮的验证准确率比之前更好，就保存当前参数。
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"训练和验证总耗时：{time_use // 60:.0f}m {time_use % 60:.0f}s")

    # 训练结束后，把模型参数恢复成“验证集表现最好”的那一版。
    model.load_state_dict(best_model_wts)

    # 把最优模型参数保存到磁盘，方便后续测试或部署。
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 用 DataFrame 整理训练过程，后续画图会更方便。
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("训练流程需要 pandas 来整理每个 epoch 的指标，请先安装 pandas。") from exc

    train_process = pd.DataFrame(
        data={
            "epoch": range(1, num_epochs + 1),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    return train_process


def matplot_acc_loss(train_process):
    """
    把训练过程中的损失和准确率画出来。
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("绘制损失和准确率曲线需要 matplotlib，请先安装 matplotlib。") from exc

    plt.figure(figsize=(12, 4))

    # 左图：训练损失和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], "bs-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 右图：训练准确率和验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], "ro-", label="Train Acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], "bs-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 创建模型
    model = GoogLeNet(Inception)

    # 2. 准备训练集和验证集
    train_dataloader, val_dataloader = train_val_data_process()

    # 3. 训练模型
    train_process = train_model_process(model, train_dataloader, val_dataloader, 50)

    # 4. 可视化训练过程
    matplot_acc_loss(train_process)
