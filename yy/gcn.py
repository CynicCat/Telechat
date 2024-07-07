import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder

# 读取数据
edges_df = pd.read_csv('trainSet_res.csv', low_memory=False)
node_labels_df = pd.read_csv('trainSet_ans-20240524203803153.csv', low_memory=False)
validation_df = pd.read_csv('validationSet_res.csv', low_memory=False)

# 提取msisdn和other_party的节点特征
msisdn_features = edges_df[['msisdn', 'home_area_code', 'visit_area_code', 'phone1_type', 'phone1_loc_city',
                            'phone1_loc_province']].drop_duplicates()
other_party_features = edges_df[['other_party', 'called_home_code', 'called_code', 'phone2_type', 'phone2_loc_city',
                                 'phone2_loc_province']].drop_duplicates()

# 合并标签
msisdn_features = msisdn_features.merge(node_labels_df[['msisdn', 'is_sa']], on='msisdn', how='left')


# 编码节点特征
def encode_features(df, columns, encoders=None):
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in columns}
    else:
        for col in columns:
            new_labels = df[col][~df[col].isin(encoders[col].classes_)]
            if not new_labels.empty:
                encoders[col].classes_ = np.concatenate([encoders[col].classes_, new_labels.unique()])
            df[col] = encoders[col].transform(df[col])
    return df, encoders


msisdn_features, msisdn_encoders = encode_features(msisdn_features, [
    'home_area_code', 'visit_area_code', 'phone1_type', 'phone1_loc_city', 'phone1_loc_province', 'is_sa'
])

other_party_features, other_party_encoders = encode_features(other_party_features, [
    'called_home_code', 'called_code', 'phone2_type', 'phone2_loc_city', 'phone2_loc_province'
])

# 确保所有特征列都是数值类型
msisdn_features = msisdn_features.apply(pd.to_numeric, errors='coerce')
msisdn_features = msisdn_features.dropna()

# 将特征转换为张量
msisdn_x = torch.tensor(msisdn_features.values[:, 1:-1].astype(np.float32), dtype=torch.float)  # 排除msisdn和is_sa列
labels = torch.tensor(msisdn_features['is_sa'].values.astype(np.float32), dtype=torch.float)

# 构建图数据
msisdn_to_index = {msisdn: idx for idx, msisdn in enumerate(msisdn_features['msisdn'])}
edges = edges_df[['msisdn', 'other_party']].apply(
    lambda x: (msisdn_to_index.get(x['msisdn'], -1), msisdn_to_index.get(x['other_party'], -1)), axis=1)
edges = edges[edges.apply(lambda x: x[0] != -1 and x[1] != -1)]  # 过滤无效边

if not edges.empty:
    edges = torch.tensor(list(edges), dtype=torch.long).t()
else:
    edges = torch.empty((2, 0), dtype=torch.long)

edge_attr = torch.tensor(edges_df[['call_duration']].values, dtype=torch.float)  # 举例用'call_duration'作为边特征

train_graph_data = Data(x=msisdn_x, edge_index=edges, edge_attr=edge_attr)


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x


# 初始化模型
model = GCN(input_dim=train_graph_data.num_node_features, hidden_dim=64, output_dim=1)

# 选择优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()  # 二分类任务的损失函数


# 训练模型
def train(model, data, labels, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# 开始训练
train(model, train_graph_data, labels, optimizer, criterion, epochs=10)


# 模型预测
def predict(model, data, threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(data)).squeeze().numpy()  # 使用sigmoid激活函数
    binary_output = (output >= threshold).astype(int)
    return binary_output, output


# 处理验证集
validation_msisdn_features = validation_df[
    ['msisdn', 'home_area_code', 'visit_area_code', 'phone1_type', 'phone1_loc_city',
     'phone1_loc_province']].drop_duplicates()
validation_msisdn_features, _ = encode_features(validation_msisdn_features, [
    'home_area_code', 'visit_area_code', 'phone1_type', 'phone1_loc_city', 'phone1_loc_province'
], msisdn_encoders)

validation_msisdn_features = validation_msisdn_features.apply(pd.to_numeric, errors='coerce')
validation_msisdn_features = validation_msisdn_features.dropna()

validation_msisdn_to_index = {msisdn: idx for idx, msisdn in enumerate(validation_msisdn_features['msisdn'])}
validation_edges = validation_df[['msisdn', 'other_party']].apply(
    lambda x: (validation_msisdn_to_index.get(x['msisdn'], -1), validation_msisdn_to_index.get(x['other_party'], -1)),
    axis=1)
validation_edges = validation_edges[validation_edges.apply(lambda x: x[0] != -1 and x[1] != -1)]  # 过滤无效边

if not validation_edges.empty:
    validation_edges = torch.tensor(list(validation_edges), dtype=torch.long).t()
else:
    validation_edges = torch.empty((2, 0), dtype=torch.long)

validation_edge_attr = torch.tensor(validation_df[['call_duration']].values, dtype=torch.float)

validation_x = torch.tensor(validation_msisdn_features.values[:, 1:].astype(np.float32), dtype=torch.float)
validation_graph_data = Data(x=validation_x, edge_index=validation_edges, edge_attr=validation_edge_attr)

# 预测验证集上的 is_sa 值
validation_binary_predictions, validation_prob_predictions = predict(model, validation_graph_data)

# 将预测结果保存到文件中，并确保msisdn值不重复
validation_msisdn_features['predicted_is_sa'] = validation_prob_predictions
validation_msisdn_features = validation_msisdn_features.groupby('msisdn', as_index=False)['predicted_is_sa'].mean()
validation_msisdn_features['predicted_is_sa'] = (validation_msisdn_features['predicted_is_sa'] >= 0.5).astype(int)

validation_msisdn_features.to_csv('validation_predicted.csv', index=False)

# 打印前几个预测结果
print(validation_msisdn_features.head(10))
