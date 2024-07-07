import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, GCNConv
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import torch.nn.functional as F


# 读取并合并训练集数据
train_set = pd.read_csv('trainSet_res.csv', dtype={'home_area_code': str, 'visit_area_code': str, 'called_code': str}).merge(
    pd.read_csv('trainSet_ans.csv'), on='msisdn', how='left'
)

# 读取验证集数据
validation_set = pd.read_csv('validationSet_res.csv', dtype={'home_area_code': str, 'visit_area_code': str, 'called_code': str})

# 纵向合并训练集和验证集
combined_set = pd.concat([train_set, validation_set], ignore_index=True)

categorical_features = ['home_area_code', 'visit_area_code', 'called_home_code', 'called_code']

# 检查多个列是否包含非数字字符
non_numeric_rows = []

for col in categorical_features:
    combined_set[f'{col}_numeric'] = pd.to_numeric(combined_set[col], errors='coerce')
    non_numeric_rows.extend(combined_set[combined_set[f'{col}_numeric'].isna()].index.tolist())

# 去重并排序索引
non_numeric_rows = sorted(set(non_numeric_rows))

combined_set = combined_set.drop(index=non_numeric_rows)
non_numeric_rows

from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 读取CSV数据
data = combined_set

# 处理类别型特征
categorical_features = ['home_area_code', 'visit_area_code']
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(data[categorical_features].astype(str)).toarray()
data[categorical_features].replace('0sc', 0)

# 转换时间字段为时间戳
data['start_time'] = pd.to_datetime(data['start_time'], format='%Y%m%d%H%M%S', errors='coerce')
data['end_time'] = pd.to_datetime(data['end_time'], format='%Y%m%d%H%M%S', errors='coerce')
data['update_time'] = pd.to_datetime(data['update_time'], format='%Y%m%d%H%M%S', errors='coerce')
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d%H%M%S', errors='coerce')
data['date_c'] = pd.to_datetime(data['date_c'], format='%Y%m%d%H%M%S', errors='coerce')
data['open_datetime'] = pd.to_datetime(data['open_datetime'], format='%Y%m%d%H%M%S', errors='coerce')

# 编码call_event字段
data['call_event'] = data['call_event'].map({'call_src': 1, 'call_dst': 0})

# 将long_type1字段转换为独热编码
data = pd.get_dummies(data, columns=['long_type1'], prefix='long_type1')

# 排除编号为21到26的字段
exclude_fields = ['phone1_type', 'phone2_type', 'phone1_loc_city', 'phone1_loc_province', 'phone2_loc_city', 'phone2_loc_province','a_product_id']

combined_set = combined_set.drop(columns=exclude_fields)

combined_set.shape

# 提取电话号码并形成图的节点
import math
msisdn = pd.concat([combined_set['msisdn'], combined_set['other_party']]).unique()
num_nodes = len(msisdn)
msisdn_to_idx = {number: idx for idx, number in enumerate(msisdn)}  # 将电话号码映射到节点索引
edges = []
labels = []
for idx, row in combined_set.iterrows():
    source = msisdn_to_idx[row['msisdn']]
    destination = msisdn_to_idx[row['other_party']]
    is_sa = row['is_sa']
    if(math.isnan(is_sa)):
        labels.append((0))
    else:
        labels.append((is_sa))
    edges.append((source, destination))
    
# 转换为张量格式
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
num_edges = edge_index.shape[1]

# 提取所有列作为节点特征
# 选择需要作为节点特征的列（排除边索引和标签列）
features_columns = [col for col in data.columns if col not in ['msisdn', 'other_party', 'is_sa','start_time', 'end_time', 'update_time', 'date', 'date_c', 'open_datetime']]
node_features = data[features_columns].values.astype(np.float32)
x = torch.tensor(node_features, dtype=torch.float)

# 处理标签（is_sa 字段作为标签）
#labels = combined_set['is_sa'].values.astype(np.float32)
labels = torch.tensor(labels, dtype=torch.long).view(-1)  # 确保标签是列向量

edge_index.size()
x.shape
num_train = train_set.shape[0] // 5 * 4
num_all_train = train_set.shape[0]
trainX = x[:num_train]
validateX = x[num_train:num_all_train]
trainY = labels[:num_train]
validateY = labels[num_train:num_all_train]

# 定义图数据
data = Data(x=trainX, edge_index=edge_index)

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # 添加批归一化
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)  # 添加批归一化
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)  # 添加批归一化

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        return x

in_channels = x.shape[1]
hidden_channels = 64  
out_channels = 2  

# 训练GCN模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = labels.to('cpu')
# 创建模型
model = GCN(in_channels, hidden_channels, out_channels).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))  # 调整类别权重

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 学习率调度器

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data).to('cpu')
    loss = criterion(out, trainY)  # 使用交叉熵损失函数
    loss.backward()
    optimizer.step()
    scheduler.step()  # 更新学习率
    return loss.item()

data_validate = Data(x=trainX,edge_index=edge_index)
def test():
    model.eval()
    with torch.no_grad():
        logits = model(data_validate).to('cpu')
        pred = torch.argmax(logits, dim=1)
        correct = pred.eq(trainY).sum().item()
        acc = correct / len(trainY)
    return acc

# 训练和测试
for epoch in range(1, 100):
    loss = train()
    acc = test()
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

print("训练完成，模型已准备好进行推理。")
# 保存模型
torch.save(model.state_dict(), 'gcn_model.pth')
print("模型已保存。")

# 随机生成节点特征
x = torch.randn((num_edges, 256), dtype=torch.float)

# 定义图数据
data = Data(x=x, edge_index=edge_index)

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        return x

# 实例化模型
in_channels = x.shape[1]
hidden_channels = 64
out_channels = 2
model = GCN(in_channels, hidden_channels, out_channels)

# 加载训练好的模型参数
model.load_state_dict(torch.load('gcn_model.pth'))

# 进行推理
model.eval()
with torch.no_grad():
    logits = model(data)
    predictions = torch.argmax(logits, dim=1)

# 将预测结果添加到验证集中
validation_set['is_sa'] = predictions.cpu().numpy()

# 输出预测结果
print(validation_set[['msisdn', 'other_party', 'is_sa']])
# 保存结果（只保留 msisdn 和 predictions 列）
result = validation_set[['msisdn', 'is_sa']]
filtered_data = result.drop_duplicates(subset='msisdn', keep='first')

filtered_data.to_csv('validationSet_predictions.csv', index=False)
