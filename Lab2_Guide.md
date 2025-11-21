# 实验二：Food-11 图像分类 Jupyter Notebook 实验指南

这份指南将一步步教你如何在 Jupyter Notebook 中完成实验二的所有要求。你可以新建一个 `.ipynb` 文件（例如命名为 `Lab2_Food11.ipynb`），然后按照下面的步骤，将代码块复制到你的 Notebook 单元格中运行。

## 第一步：环境设置与导入库

首先，我们需要导入实验所需的 Python 库，并设置随机种子以保证结果可复现。

**在第一个单元格中输入以下代码：**

```python
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义数据集路径 (根据你提供的信息)
# 使用 os.path.abspath 确保路径格式统一
DATA_DIR = os.path.abspath("d:/study/深度学习/实验二 CNN/dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "training")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "evaluation")

print(f"Training dir: {TRAIN_DIR}")
```

---

## 第二步：定义数据集类 (Dataset)

**注意**：由于你的路径中包含中文（`深度学习`），OpenCV 的 `cv2.imread` 在 Windows 上无法直接读取。我们需要定义一个辅助函数 `cv_imread` 来解决这个问题。

**新建一个单元格，输入以下代码：**

```python
def cv_imread(file_path):
    """
    使用 cv2.imdecode 读取包含中文路径的图片
    """
    try:
        # np.fromfile 可以正确读取中文路径
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

class FoodDataset(Dataset):
    def __init__(self, image_path, image_size=(128, 128), mode='train', transforms=None):
        self.image_path = image_path
        # 读取目录下所有图片文件
        self.image_file_list = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.mode = mode
        self.image_size = image_size
        self.transforms = transforms
        
        # 如果是训练或验证模式，需要提取标签
        if mode in ['train', 'val']:
            self.labels = []
            for filename in self.image_file_list:
                # 文件名格式: 3_123.jpg -> 标签为 3
                try:
                    label = int(filename.split("_")[0])
                    self.labels.append(label)
                except:
                    print(f"Warning: Could not parse label from {filename}")
                    self.labels.append(0) # 默认标签

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        img_name = self.image_file_list[idx]
        img_path = os.path.join(self.image_path, img_name)
        
        # 使用自定义的 cv_imread 读取图片 (解决中文路径问题)
        img = cv_imread(img_path)
        
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
        img = cv2.resize(img, self.image_size)      # 调整大小
        
        # 应用数据增强或预处理
        if self.transforms:
            img = self.transforms(img)
        else:
            # 默认转换为 Tensor 并归一化
            img = transforms.ToTensor()(img)
        
        if self.mode in ['train', 'val']:
            label = self.labels[idx]
            return img, label
        else:
            return img
```

---

## 第三步：数据增强 (Data Augmentation)

这是实验的得分点之一。我们需要设计 `train_transform`，至少包含 5 种变换。

**新建一个单元格，输入以下代码：**

```python
# 训练集数据增强 (至少 5 种)
def get_train_transforms(image_size=(128, 128)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        # 1. 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 2. 随机旋转
        transforms.RandomRotation(degrees=15),
        # 3. 颜色抖动 (亮度、对比度、饱和度、色调)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 4. 随机裁剪
        transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
        # 5. 随机仿射变换
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        transforms.ToTensor(),
        # 归一化 (使用 ImageNet 的均值和方差)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 验证/测试集预处理 (不进行增强，只进行 Resize 和 Normalize)
def get_test_transforms(image_size=(128, 128)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

---

## 第四步：可视化数据增强效果

我们需要展示数据增强前后的对比，这也是得分点。

**新建一个单元格，输入以下代码：**

```python
def visualize_augmentations(dataset, num_samples=5):
    """可视化原图和增强后的图片"""
    # 创建一个没有增强的 dataset 用于对比
    base_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    orig_dataset = FoodDataset(dataset.image_path, mode=dataset.mode, transforms=base_transforms)
    
    plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # 获取原图
        orig_img, label = orig_dataset[i]
        # 转换为 numpy 格式 (C, H, W) -> (H, W, C)
        orig_img = orig_img.permute(1, 2, 0).numpy()
        
        # 获取增强后的图
        aug_img, _ = dataset[i]
        aug_img = aug_img.permute(1, 2, 0).numpy()
        # 反归一化以便显示
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        aug_img = aug_img * std + mean
        aug_img = np.clip(aug_img, 0, 1)
        
        # 显示原图
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(orig_img)
        plt.title(f"Original (Label: {label})")
        plt.axis('off')
        
        # 显示增强图
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(aug_img)
        plt.title("Augmented")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# 实例化数据集并可视化
# 确保路径存在
if os.path.exists(TRAIN_DIR):
    train_dataset = FoodDataset(TRAIN_DIR, mode='train', transforms=get_train_transforms())
    visualize_augmentations(train_dataset)
else:
    print(f"Error: Directory not found: {TRAIN_DIR}")
```

---

## 第五步：搭建模型

这里我们需要定义两个模型：一个是自定义的 CNN，一个是 VGG 模型。

**新建一个单元格，输入以下代码：**

```python
# 1. 自定义 CNN 模型 (升级为 ResNet-18 以达到更高准确率)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11):
        super(ResNet, self).__init__()
        self.in_planes = 32 # 减小初始通道数 (64 -> 32)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def CustomCNN(num_classes=11):
    # 使用 Light ResNet-18 (通道数减半)
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# 2. VGG 模型 (使用 torchvision)
def get_vgg_model(num_classes=11):
    # 加载预训练的 VGG16
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    # 冻结前面的层 (可选，如果想微调可以不冻结)
    # for param in model.features.parameters():
    #     param.requires_grad = False
        
    # 修改最后的全连接层以匹配我们的类别数 (11类)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model

# 打印模型参数量
my_model = CustomCNN().to(device)
# vgg_model = get_vgg_model().to(device)

print(f"Custom LightResNet-18 Parameters: {sum(p.numel() for p in my_model.parameters()):,}")
# print(f"VGG16 Parameters: {sum(p.numel() for p in vgg_model.parameters()):,}")
```

---

## 第六步：训练函数与 TensorBoard

我们需要编写一个通用的训练函数，并集成 TensorBoard 来记录 Loss 和 Accuracy。

**新建一个单元格，输入以下代码：**

```python
def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, name="model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 使用 CosineAnnealingLR 学习率调度器，更平滑地调整学习率，有助于模型收敛到更好的局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # TensorBoard Writer
    writer = SummaryWriter(f'runs/{name}')
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 写入 TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{name}_best.pth')
            
        scheduler.step()
        
    writer.close()
    print(f"Training Complete. Best Val Acc: {best_acc:.2f}%")
    return model
```

---

## 第七步：开始训练

现在我们可以准备 DataLoader 并开始训练了。

**新建一个单元格，输入以下代码：**

```python
# 准备 DataLoader
BATCH_SIZE = 32 # 减小 Batch Size 以节省显存

train_dataset = FoodDataset(TRAIN_DIR, mode='train', transforms=get_train_transforms())
val_dataset = FoodDataset(VAL_DIR, mode='val', transforms=get_test_transforms())

# --- 数据验证 (Sanity Check) ---
print("=== Data Verification ===")
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Val Dataset Size: {len(val_dataset)}")
# 打印前 5 个样本的文件名和标签，检查是否对应正确
for i in range(5):
    filename = train_dataset.image_file_list[i]
    label = train_dataset.labels[i]
    print(f"File: {filename} -> Label: {label}")
print("=========================\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Windows下 num_workers 建议设为 0
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# 加载最佳权重进行评估
# 注意：如果你训练的是 ResNet，文件名应该是 'custom_resnet_best.pth'
my_model.load_state_dict(torch.load('custom_resnet_best.pth'))
plot_confusion_matrix(my_model, val_loader, "Custom ResNet")
```

---

## 第九步：生成测试集预测结果 (CSV)

最后，我们需要对 `evaluation` 文件夹中的图片进行预测，并生成 CSV 文件。

**新建一个单元格，输入以下代码：**

```python
def generate_predictions(model, test_dir, output_file):
    # 测试集 Dataset (mode='test')
    test_dataset = FoodDataset(test_dir, mode='test', transforms=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            
            # 获取对应的文件名
            start_idx = i * 32
            end_idx = start_idx + len(inputs)
            batch_filenames = test_dataset.image_file_list[start_idx:end_idx]
            filenames.extend(batch_filenames)
            
    # 创建 DataFrame 并保存
    df = pd.DataFrame({
        'Id': filenames,
        'Category': predictions
    })
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

完成以上步骤后，你将得到：
1.  `custom_cnn_best.pth` 和 `vgg16_best.pth`: 训练好的模型权重。
2.  `runs/` 文件夹: 包含 TensorBoard 日志（可以在终端运行 `tensorboard --logdir=runs` 查看）。
3.  `ans_ours.csv` 和 `ans_vgg.csv`: 需要提交的预测结果文件。
4.  Notebook 中展示的各种可视化图表（数据增强、混淆矩阵）。

---

## 额外步骤：解决过拟合并冲刺 0.8 (Fine-tuning v2)

**你的训练日志显示 Train Acc (85%) > Val Acc (78%)，确实出现了过拟合。**
我们采用以下策略：
1.  加载之前最好的模型 (epoch 43 达到了 78.37%)
2.  **增加 Dropout** (在全连接层前插入，强力抑制过拟合)
3.  使用更强的正则化 (Weight Decay) 和更小的学习率

**新建一个单元格，输入以下代码并运行：**

```python
print("\n=== Fine-tuning with Regularization ===")

# 1. 重新定义带 Dropout 的全连接层
# 注意：这里我们动态修改已加载模型的最后一层，插入 Dropout
my_model = CustomCNN().to(device)
# 加载之前训练最好的权重
my_model.load_state_dict(torch.load('custom_resnet_best.pth'))

# 在最后一层线性层前插入 Dropout
# ResNet 的结构是: ... -> layer4 -> avgpool -> linear
# 我们把 linear 替换为一个包含 Dropout 的 Sequential
num_ftrs = my_model.linear.in_features
my_model.linear = nn.Sequential(
    nn.Dropout(p=0.5), # 50% 的概率丢弃神经元，强力抑制过拟合
    nn.Linear(num_ftrs, 11)
).to(device)

# 2. 使用更小的学习率和更大的权重衰减 (Weight Decay)
# weight_decay 增大到 1e-3，进一步限制参数大小
optimizer = optim.Adam(my_model.parameters(), lr=0.0001, weight_decay=1e-3) 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

criterion = nn.CrossEntropyLoss()

# 再训练 20 个 epoch
best_acc = 0.0
# 尝试读取之前的最佳准确率
try:
    # 这里只是为了打印，实际逻辑以 val_acc > best_acc 为准
    print("Loaded previous best model weights.")
except:
    pass

for epoch in range(20):
    my_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    
    my_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = my_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_acc = 100. * correct / total
    print(f"Fine-tune Epoch [{epoch+1}/20] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # 保存微调后的最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(my_model.state_dict(), 'custom_resnet_finetune_best.pth')

print(f"Fine-tuning Complete. Best Val Acc: {best_acc:.2f}%")

# 3. 评估微调后的模型
my_model.load_state_dict(torch.load('custom_resnet_finetune_best.pth'))
plot_confusion_matrix(my_model, val_loader, "Custom ResNet (Fine-tuned)")

# 4. 生成最终预测结果 (覆盖之前的 ans_ours.csv)
generate_predictions(my_model, TEST_DIR, 'ans_ours.csv')
```
