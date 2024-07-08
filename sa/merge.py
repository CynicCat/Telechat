import pandas as pd
import numpy as np
# 读取第一个CSV文件（包含标签的表）
labels_df = pd.read_csv('trainSet_ans.csv',encoding='utf-8')

# 读取个CSV文件（没有标签的表）
data_df = pd.read_csv('trainSet_res.csv',encoding='utf-8')

labels_df = labels_df.drop_duplicates(subset='msisdn')#去重复项

labels_df['msisdn'] = labels_df['msisdn'].astype(str)
data_df['msisdn'] = data_df['msisdn'].astype(str)

merged_df = pd.merge(data_df, labels_df, on='msisdn',how='left')# 合并两个数据框架，按msisdn进行合并

wa = merged_df.columns.get_loc('date_c') + 1
wb = merged_df.columns.get_loc('is_sa')#找索引
merged_df = merged_df.drop(merged_df.columns[wa:wb], axis=1)#删掉没用的

# 将合后的数据框架保存到新的CSV文件
merged_df.to_csv('merged_data.csv',index=False)

