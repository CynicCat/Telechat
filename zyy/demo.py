import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 读取数据集
train_set_ans = pd.read_csv('../LLM/data/trainSet_ans.csv')
train_set_res = pd.read_csv('../LLM/data/trainSet_res.csv', low_memory=False)
validation_data = pd.read_csv('../LLM/data/validationSet_res.csv')

# 合并训练数据
train_data = pd.merge(train_set_ans, train_set_res, on='msisdn')

# 数据预处理函数
def preprocess_data(data, top_5_risk_rate=None, is_training=True):
    # 填充缺失值
    data = data.ffill().bfill()

    # 定义一个正则表达式模式，用于匹配中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    # 使用正则表达式过滤出只包含中文字符的行（验证集保留所有行）
    if is_training:
        data = data[data['phone1_loc_province'].astype(str).apply(lambda x: chinese_pattern.search(x) is not None)]
        data = data[data['phone2_loc_province'].astype(str).apply(lambda x: chinese_pattern.search(x) is not None)]

    if is_training:
        # 过滤高风险用户
        high_risk_users = data[data['is_sa'] == 1]

        # 统计各省份高风险用户数
        src_high_risk_counts = high_risk_users[high_risk_users['call_event'] == 'call_src']['phone1_loc_province'].value_counts()
        dst_high_risk_counts = high_risk_users[high_risk_users['call_event'] == 'call_dst']['phone2_loc_province'].value_counts()

        # 统计各省份总用户数
        src_total_counts = data[data['call_event'] == 'call_src']['phone1_loc_province'].value_counts()
        dst_total_counts = data[data['call_event'] == 'call_dst']['phone2_loc_province'].value_counts()

        # 计算风险率
        src_risk_rate = src_high_risk_counts / src_total_counts
        dst_risk_rate = dst_high_risk_counts / dst_total_counts

        # 合并结果
        risk_rate = pd.concat([src_risk_rate, dst_risk_rate], axis=1).fillna(0)
        risk_rate.columns = ['src_risk_rate', 'dst_risk_rate']

        # 计算综合风险率
        risk_rate['combined_risk_rate'] = risk_rate['src_risk_rate'] + risk_rate['dst_risk_rate']

        # 排序并选择前五个省份
        top_5_risk_rate = risk_rate.nlargest(5, 'combined_risk_rate')

    # 二值化处理
    data['binary_risk_rate'] = np.where(
        data['phone1_loc_province'].isin(top_5_risk_rate.index) | 
        data['phone2_loc_province'].isin(top_5_risk_rate.index), 1, 0)

    # 对call_duration进行最小-最大归一化
    min_max_scaler = MinMaxScaler()
    data['call_duration'] = min_max_scaler.fit_transform(data[['call_duration']])

    # 对lfee数据处理
    data['lfee'] = data['lfee'].apply(lambda x: max(x, 0))
    data['lfee'] = data['lfee'].apply(lambda x: np.log1p(x))
    data['lfee'] = min_max_scaler.fit_transform(data[['lfee']])

    # 对cfee数据处理
    data['cfee'] = data['cfee'].apply(lambda x: np.log1p(x))
    data['cfee'] = min_max_scaler.fit_transform(data[['cfee']])

    # 表示号码是否离开归属地区，为1表示未离开
    data['home_area_code'] = data['home_area_code'].astype(str).str.lstrip('0')
    data['visit_area_code'] = data['visit_area_code'].astype(str).str.lstrip('0')
    data['called_home_code'] = data['called_home_code'].astype(str).str.rstrip('0').str.replace('.', '')
    data['called_code'] = data['called_code'].astype(str).str.lstrip('0')
    data['code'] = (data['home_area_code'] == data['visit_area_code']).astype(int)
    data['c_code'] = (data['called_home_code'] == data['called_code']).astype(int)

    # 识别call_event为'call_dst'的行
    call_dst_mask = data['call_event'] == 'call_dst'

    # 交换'msisdn'和'other_party'的值
    data.loc[call_dst_mask, ['msisdn', 'other_party']] = data.loc[call_dst_mask, ['other_party', 'msisdn']].values

    # 交换'c_code'和'called_code'的值
    data.loc[call_dst_mask, ['c_code', 'code']] = data.loc[call_dst_mask, ['code', 'c_code']].values

    # 去掉无关的数据项 
    data = data.drop(columns=['phone1_loc_province', 'phone2_loc_province', 'phone1_loc_city', 'phone2_loc_city', 'ismultimedia',
                                          'update_time', 'date_c', 'a_serv_type', 'end_time', 'home_area_code', 'visit_area_code', 'called_home_code',
                                            'called_code','call_event', 'phone1_type', 'phone2_type' ,'start_time'])

    # 定义映射关系
    long_type1_map = {0: 'LT0', 1: 'LT1', 2: 'LT2', 3: 'LT3'}
    roam_type_map = {0: 'RT0', 1: 'RT1', 4: 'RT4', 6: 'RT6', 8: 'RT8', 5: 'RT5'}

    # 执行映射
    data['long_type1'] = data['long_type1'].map(long_type1_map)
    data['roam_type'] = data['roam_type'].map(roam_type_map)

    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.strftime('%Y%m%d')
    data['date'] = data['date'].astype('int64')

    # 使用One-Hot编码处理分类变量
    data = pd.get_dummies(data)

    return data, top_5_risk_rate

# 预处理训练数据并计算高风险省份
train_data_preprocessed, top_5_risk_rate = preprocess_data(train_data)

# 分离特征和标签
X_train = train_data_preprocessed.drop('is_sa', axis=1)
y_train = train_data_preprocessed['is_sa']

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 训练随机森林模型
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 保存原始验证数据的索引
original_validation_index = validation_data.index

# 预处理验证数据（使用计算出的高风险省份信息）
validation_data_preprocessed, _ = preprocess_data(validation_data, top_5_risk_rate, is_training=False)

# 验证集预处理后的数据可能包含新的列，确保与训练集的列一致
missing_cols = set(X_train.columns) - set(validation_data_preprocessed.columns)
for col in missing_cols:
    validation_data_preprocessed[col] = 0

extra_cols = set(validation_data_preprocessed.columns) - set(X_train.columns)
validation_data_preprocessed = validation_data_preprocessed.drop(columns=extra_cols)

validation_data_preprocessed = validation_data_preprocessed[X_train.columns]

# 在验证集上进行预测
validation_predictions = model.predict(validation_data_preprocessed)

# 添加预测结果到验证数据
validation_data['is_sa'] = validation_predictions

# 打印验证集上的预测结果
print(validation_data[['msisdn', 'is_sa']])

# 导出预测结果
validation_data[['msisdn', 'is_sa']].to_csv('../LLM/data/predicted_validation_results.csv', index=False)
print("预测结果已导出为CSV文件。")
