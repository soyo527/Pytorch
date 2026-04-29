from pathlib import Path
from unittest import result

import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder

from model import GoogLeNet, Inception
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "test"
MODEL_SAVE_PATH = PROJECT_ROOT / "model"
MODEL_SAVE_PATH = PROJECT_ROOT / "best_model.pth"


def test_data_process():
    """
    读取 FashionMNIST 官方测试集。

    train=False 表示这里拿到的是测试集，不参与训练，
    专门用来检验模型的泛化能力。
    """

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.162, 0.151, 0.138],
            std=[0.058, 0.052, 0.048]
        ),
    ])

    test_data = ImageFolder(DATA_ROOT, transform=test_transform)

    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return test_dataloader


def test_model_process(model, test_dataloader):
    """
    计算模型在整个测试集上的准确率。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_corrects = 0
    test_num = 0

    # 测试阶段不更新参数，所以要关闭梯度计算。
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y)
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print(f"测试集准确率为：{test_acc:.4f}")


if __name__ == "__main__":
    # 1. 创建和训练时相同结构的模型
    model = GoogLeNet(Inception)

    # 2. 加载训练阶段保存下来的最佳参数
    # map_location='cpu' 的作用是：
    # 即使这份模型最初在 GPU 上训练，也能在只有 CPU 的机器上顺利加载。
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    classes = ['Cat', 'Dog']

    image = Image.open(PROJECT_ROOT / "maodie.png")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.162, 0.151, 0.138],
            std=[0.058, 0.052, 0.048]
        ),
    ])

    image = test_transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：", classes[result])