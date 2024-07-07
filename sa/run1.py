import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
from fct import convert_to_numeric,clean_and_aggregate_features


df1 = pd.read_csv(r'F:\pythonProject1\.venv\Scripts\merged_data.csv')
df2 = pd.read_csv(r'F:\validationSet_res.csv')
df=df1.copy()
df.drop(['start_time', 'end_time', 'open_datetime', 'update_time', 'date', 'date_c','other_party'], axis=1, inplace=True)
df2.drop(['start_time', 'end_time', 'open_datetime', 'update_time', 'date', 'date_c','other_party'], axis=1, inplace=True)
df['call_event']=(df['call_event']=="call_src").astype(int)
df['msisdn']=df['msisdn'].astype(str)#标识号1转化为字符串

df2['call_event']=(df2['call_event']=="call_src").astype(int)
df2['msisdn']=df2['msisdn'].astype(str)#同上，对于验证集


numeric_features=['call_duration','cfee','lfee','hour']
categorical_features=['call_event', 'a_serv_type', 'long_type1', 'roam_type', 'a_product_id', 'dayofweek','phone1_type','phone2_type','phone1_loc_province','phone2_loc_province']


for column in numeric_features:
    convert_to_numeric(df,column)
    convert_to_numeric(df2, column)


for column in df.columns:
    if column in numeric_features:
        mean=df[column].mean()
        df[column].fillna(mean)
        df2[column].fillna(mean)
    if column in categorical_features:
        # 获取该列的非 NaN 众数
        most_frequent = df[column].value_counts().idxmax()
        # 填充缺失值
        df[column]=df[column].fillna(most_frequent)
        df2[column]=df2[column].fillna(most_frequent)


aggregated_data= clean_and_aggregate_features(df, numeric_features, categorical_features)
aggregated_data2= clean_and_aggregate_features(df2, numeric_features, categorical_features)

# 提取标签列
labels = df[['msisdn', 'is_sa']].drop_duplicates().set_index('msisdn')
aggregated_data = aggregated_data.set_index('msisdn').join(labels)
aggregated_data2 = aggregated_data2.set_index('msisdn')
aggregated_data = aggregated_data.dropna()


class TelecomFraudDataset(Dataset):
    def __init__(self, df, has_labels=True):
        self.df = df
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.has_labels:
            features = row.drop('is_sa').values
            label = row['is_sa']
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            features = row.values
            return torch.tensor(features, dtype=torch.float32)


dataset = aggregated_data
dataset2 = aggregated_data2
# 数值型特征标准化
scaler = StandardScaler()
dataset[numeric_features] = scaler.fit_transform(dataset[numeric_features])
dataset2[numeric_features] = scaler.fit_transform(dataset2[numeric_features])

# 类别型特征编码
for feature in categorical_features:
    le = LabelEncoder()
    dataset[feature] = le.fit_transform(dataset[feature])
    dataset2[feature] = le.fit_transform(dataset2[feature])



train_df = dataset
val_df = dataset2

# 创建数据集和数据加载器
train_dataset = TelecomFraudDataset(train_df)
val_dataset = TelecomFraudDataset(val_df, has_labels=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 初始化模型
input_dim = len(dataset.columns) - 1  # 减去目标列
model = FraudDetectionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
for epoch in range(200):  # 设置训练的epoch数量
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
val_outputs = []
with torch.no_grad():
    for i, features in enumerate(val_loader):
        outputs = model(features).squeeze()
        val_outputs.extend(outputs.tolist())

val_outputs = [1 if x >= 0.5 else 0 for x in val_outputs]
val_msisdns = val_df.index.tolist()
# 将val_labels和val_outputs转化为DataFrame
results_df = pd.DataFrame({'msisdn': val_msisdns, 'is_sa': val_outputs})

# 保存预测结果
results_df.to_csv('F:/pythonProject1/.venv/Scripts/results.csv', index=False)

# 保存模型
torch.save(model.state_dict(), 'fraud_detection_model.pth')