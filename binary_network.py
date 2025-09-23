import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from torch.cuda.amp import autocast
import os
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

# ============== 配置区域 ==============
IMG_SIZE = 64   # 输入维度（你可以改成 256, 384, 512 ...）
BATCH_SIZE = 8   # batch 大小，受显存限制
EPOCHS = 5       # 训练轮数
LR = 1e-4        # 初始学习率
MODEL_NAME = "vit_huge_patch14_224"  # timm 模型
NUM_CLASSES = 2  # 二分类
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== 示例数据集（你需要替换成自己的） ==============
class RandomDataset(Dataset):
    """一个假的数据集（随机图片 + 标签），仅用于测试代码能跑通"""
    def __init__(self, length=1000, img_size=IMG_SIZE):
        self.length = length
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(4, IMG_SIZE, IMG_SIZE)  # 随机生成假图片
        label = torch.randint(0, 2, (1,)).item()  # 随机 0/1
        return img, label

#train_dataset = RandomDataset(500)
val_dataset = RandomDataset(100)
class MyDataset(Dataset):
    def __init__(self, data_dict, label=1):
        self.samples = []
        for element in data_dict:
            for k1, v1 in element.items():        # 一级 key (prompt)
                for k2, v2 in v1.items():           # 二级 key (step)
                    # v2 是你真正需要的张量
                    if isinstance(v2, torch.Tensor):
                        self.samples.append((v2, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, label = self.samples[idx]

    # 如果是 (1, C, H, W)，去掉多余的维度
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)  # (C, H, W)

        return x, label
train_dataset=torch.load('intermediate.pt')
train_dataset1=torch.load('intermediate_unsafe.pt')
train_dataset=MyDataset(train_dataset,label=0)
train_dataset1=MyDataset(train_dataset1,label=1)
train_dataset = ConcatDataset([train_dataset, train_dataset1])
total_len=len(train_dataset)
train_len=int(0.85*total_len)
val_len=total_len-train_len
train_dataset,val_dataset=random_split(train_dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ============== 定义模型 ==============
class BinaryClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

model = BinaryClassifier().to(DEVICE)
print(f"Total Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ============== 优化器 & 损失函数 ==============
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ============== 训练 & 验证 ==============
best_acc = 0.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
        reserved=imgs[:,3,:,:]
        imgs = imgs[:, :3, :, :]
        imgs[:,0, :, :]+=reserved/3
        imgs[:,1, :, :]+=reserved/3
        imgs[:,2, :, :]+=reserved/3
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        print("train"+str(imgs.shape))
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    train_loss /= len(train_dataset)

    # ---- Val ----
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):

            imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
            imgs = imgs[:, :3, :, :]
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    # ---- Save best ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"checkpoints/best_model.pth")
        print(">> Saved best model")

print("Training done! Best Val Acc:", best_acc)
