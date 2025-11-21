import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# 建议将数据集路径修改为相对路径或通过参数传入
TRAIN_DIR = r'd:/study/深度学习/实验二 CNN/dataset/training'
VAL_DIR = r'd:/study/深度学习/实验二 CNN/dataset/validation'
TEST_DIR = r'd:/study/深度学习/实验二 CNN/dataset/evaluation'
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Handling (with Chinese Path Fix) ---
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
        self.image_size = image_size
        self.mode = mode
        self.transforms = transforms
        self.image_file_list = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
        
        if mode != 'test':
            self.labels = [int(f.split("_")[0]) for f in self.image_file_list]
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        img_name = self.image_file_list[idx]
        img_path = os.path.join(self.image_path, img_name)
        
        # 使用自定义的 cv_imread 读取图片
        img = cv_imread(img_path)
        
        if img is None:
            # 如果读取失败，返回一个全黑图片防止程序崩溃 (或者抛出异常)
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            print(f"Warning: Could not read image {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        if self.transforms:
            img = self.transforms(img)
            
        if self.mode != 'test':
            label = self.labels[idx]
            return img, label
        else:
            return img

def get_train_transforms(image_size=(128, 128)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms(image_size=(128, 128)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- 2. Model Architecture (LightResNet-18) ---
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
    def __init__(self, block, num_blocks, num_classes=11, dropout_prob=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 32 # Light version: start with 32 channels

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        
        # Integrated Dropout in the classifier
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(256*block.expansion, num_classes)
        )

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

def CustomResNet(num_classes=11):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def get_vgg_model(num_classes=11):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

# --- 3. Training & Evaluation ---
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, name="model"):
    criterion = nn.CrossEntropyLoss()
    # Increased weight_decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # CosineAnnealingLR for smooth learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    writer = SummaryWriter(f'runs/{name}')
    best_acc = 0.0
    
    print(f"Starting training for {name}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
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
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{name}_best.pth')
            
        scheduler.step()
        
    writer.close()
    print(f"Training Complete. Best Val Acc: {best_acc:.2f}%")
    return model

def generate_predictions(model, test_dir, output_file):
    test_dataset = FoodDataset(test_dir, mode='test', transforms=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    filenames = []
    
    print(f"Generating predictions for {output_file}...")
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + len(inputs)
            batch_filenames = test_dataset.image_file_list[start_idx:end_idx]
            filenames.extend(batch_filenames)
            
    df = pd.DataFrame({'Id': filenames, 'Category': predictions})
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

# --- 4. Main Execution ---
if __name__ == '__main__':
    # Datasets & Loaders
    train_dataset = FoodDataset(TRAIN_DIR, mode='train', transforms=get_train_transforms())
    val_dataset = FoodDataset(VAL_DIR, mode='val', transforms=get_test_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 1. Train Custom ResNet
    resnet_model = CustomResNet().to(DEVICE)
    train_model(resnet_model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, name="custom_resnet")
    
    # Generate ResNet Predictions
    resnet_model.load_state_dict(torch.load('custom_resnet_best.pth'))
    generate_predictions(resnet_model, TEST_DIR, 'ans_ours.csv')
    
    # 2. Train VGG (Optional, uncomment to run)
    # print("\n=== Training VGG16 ===")
    # vgg_model = get_vgg_model().to(DEVICE)
    # train_model(vgg_model, train_loader, val_loader, num_epochs=10, lr=0.0001, name="vgg16")
    # vgg_model.load_state_dict(torch.load('vgg16_best.pth'))
    # generate_predictions(vgg_model, TEST_DIR, 'ans_vgg.csv')