import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm

# ================== 配置区域 ==================
IMG_SIZE = 224    # 输入图片尺寸，可随意修改 (≥32)
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
MODEL_NAME = "resnet50"    # ✅ 可改为 "convnext_base"、"resnet101"、"efficientnet_b4"
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 数据集 ==================
class RandomDataset(Dataset):
    """用于测试代码能否跑通的随机数据集"""
    def __init__(self, length=1000, img_size=IMG_SIZE):
        self.length = length
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(4, IMG_SIZE, IMG_SIZE)
        label = torch.randint(0, 2, (1,)).item()
        return img, label


class MyDataset(Dataset):
    """将字典形式的数据转为 (Tensor, label) 列表"""
    def __init__(self, data_dict, label=1):
        self.samples = []
        for element in data_dict:
            for k1, v1 in element.items():
                for k2, v2 in v1.items():
                    if(k2>600):
                        continue
                    if isinstance(v2, torch.Tensor):
                        self.samples.append((v2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, label = self.samples[idx]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x, label


# ⚠️ 示例加载部分，请替换为你自己的数据路径
# train_dataset = RandomDataset(500)
# val_dataset = RandomDataset(100)
train_dataset = torch.load('intermediate.pt')
train_dataset1 = torch.load('intermediate_unsafe.pt')
train_dataset = MyDataset(train_dataset, label=0)
train_dataset1 = MyDataset(train_dataset1, label=1)
train_dataset = ConcatDataset([train_dataset, train_dataset1])

# 划分训练集 / 验证集
total_len = len(train_dataset)
train_len = int(0.85 * total_len)
val_len = total_len - train_len
train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ================== 定义 CNN 模型 ==================
class BinaryClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        # 使用 timm 的 CNN backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,        # 移除原分类头
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


# ================== 初始化 ==================
model = BinaryClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
print(f"✅ Model {MODEL_NAME} loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler()  # 混合精度梯度放缩器

# ================== 训练 ==================
best_acc = 0.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    train_loss, train_correct = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        # 调整输入尺寸并融合通道
        imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        reserved = imgs[:, 3, :, :]
        imgs = imgs[:, :3, :, :]
        imgs[:, 0, :, :] += reserved / 3
        imgs[:, 1, :, :] += reserved / 3
        imgs[:, 2, :, :] += reserved / 3
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with autocast():  # 混合精度区域
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # 梯度缩放反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    train_loss /= len(train_dataset)

    # ---- Validation ----
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            reserved = imgs[:, 3, :, :]
            imgs = imgs[:, :3, :, :]
            imgs[:, 0, :, :] += reserved / 3
            imgs[:, 1, :, :] += reserved / 3
            imgs[:, 2, :, :] += reserved / 3
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
          f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # ---- Save best ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("✅ Saved best model")

print(f"🎯 Training done! Best Validation Accuracy: {best_acc:.4f}")
