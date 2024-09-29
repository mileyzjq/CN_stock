import polars as pl

# 假设A股特征数据和标签数据分别存储在 "features.csv" 和 "labels.csv"
features = pl.read_csv("features.csv")
labels = pl.read_csv("labels.csv")

# 如果需要，可以对数据进行预处理，比如缺失值填补、标准化等
# 填补缺失值示例
features = features.fill_none(0)

# 将数据转换为Pytorch张量
import torch

# 假设标签和特征都是数值类型，需要转成torch张量
X = torch.tensor(features.to_numpy(), dtype=torch.float32)
y = torch.tensor(labels.to_numpy(), dtype=torch.float32)

def rolling_train_valid_split(data, labels, window_size, step_size):
    # 滚动窗口：每次产生一个新的训练集和验证集
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        train_data = data[:end]
        train_labels = labels[:end]
        valid_data = data[end:end + step_size]
        valid_labels = labels[end:end + step_size]
        yield train_data, train_labels, valid_data, valid_labels

import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出1个节点用于二分类问题
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid函数输出二分类概率
        return x

import torch.optim as optim

def train_and_evaluate_model(X, y, window_size, step_size, epochs=10):
    input_size = X.shape[1]  # 获取输入的特征数
    model = SimpleNN(input_size)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类的交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 使用滚动窗口进行训练和验证
    for train_data, train_labels, valid_data, valid_labels in rolling_train_valid_split(X, y, window_size, step_size):
        for epoch in range(epochs):
            model.train()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels.view(-1, 1))  # 调整label形状为列向量
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            # 评估模型在验证集上的表现
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_data)
                val_loss = criterion(val_outputs, valid_labels.view(-1, 1))
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# 假设我们使用长度为100的滚动窗口，步长为10
train_and_evaluate_model(X, y, window_size=100, step_size=10)

import matplotlib.pyplot as plt

def plot_loss(train_losses, valid_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# 可以在每次训练完成后，存储训练损失和验证损失并绘制曲线



