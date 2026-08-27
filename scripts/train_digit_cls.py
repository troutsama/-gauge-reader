"""
训练轻量数字分类器 (11类: 0-9 + 小数点)
替代 EasyOCR, 适配 ARM 边缘部署
"""
import sys, random, math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path(r"D:\揭榜挂帅\gauge-reader\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = list("0123456789.")
NUM_CLASSES = len(CLASSES)  # 11

IMG_SIZE = 32


def generate_digit(label_idx, font, scale, thickness):
    """生成单个数字图像"""
    ch = CLASSES[label_idx]
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    # 居中
    x = (IMG_SIZE - tw) // 2
    y = (IMG_SIZE + th) // 2
    cv2.putText(img, ch, (x, y), font, scale, (0, 0, 0), thickness)
    return img


def augment(img):
    """增强: 旋转/缩放/模糊/噪声"""
    # 随机旋转
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)
    # 随机缩放 (通过仿射)
    s = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), 0, s)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)
    # 模糊
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    # 噪声
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(5, 20), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def generate_dataset(n_per_class=1500):
    """生成训练数据"""
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN]
    X, y = [], []
    for label_idx in range(NUM_CLASSES):
        for _ in range(n_per_class):
            font = random.choice(fonts)
            scale = random.uniform(0.8, 1.8)
            thickness = random.choice([1, 2, 3])
            img = generate_digit(label_idx, font, scale, thickness)
            img = augment(img)
            X.append(img)
            y.append(label_idx)
    X = np.array(X, dtype=np.float32) / 255.0
    X = X[:, None, :, :]  # (N, 1, 32, 32)
    y = np.array(y)
    return X, y


class TinyCNN(nn.Module):
    """极简 CNN, 参数量 ~几十KB"""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    print("生成合成数字数据集...")
    X, y = generate_dataset(n_per_class=1500)
    print(f"  数据: {X.shape}, 标签: {y.shape}")

    # 划分 train/val
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.9)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")

    model = TinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X[train_idx]).to(device)
    y_t = torch.from_numpy(y[train_idx]).to(device)
    X_v = torch.from_numpy(X[val_idx]).to(device)
    y_v = torch.from_numpy(y[val_idx]).to(device)

    batch = 128
    for epoch in range(30):
        model.train()
        total_loss = 0
        for i in range(0, len(X_t), batch):
            xb = X_t[i:i+batch]; yb = y_t[i:i+batch]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 验证
        model.eval()
        with torch.no_grad():
            acc = (model(X_v).argmax(1) == y_v).float().mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/30  loss={total_loss/len(X_t):.4f}  val_acc={acc:.4f}")

    # 保存
    model.eval()
    save_path = OUT_DIR / "digit_cls.pt"
    torch.save(model.state_dict(), str(save_path))
    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ 模型已保存: {save_path}")
    print(f"  参数量: {n_params:,}  ({n_params*4/1024:.0f} KB)")

    # 保存类别映射
    import json
    with open(OUT_DIR / "digit_classes.json", 'w') as f:
        json.dump(CLASSES, f)

    # 导出 ONNX
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = OUT_DIR / "digit_cls.onnx"
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"  ONNX: {onnx_path}")


if __name__ == '__main__':
    main()
