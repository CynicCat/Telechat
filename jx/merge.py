import pandas as pd
import numpy as np  
import networkx as nx  
import torch  
from torch_geometric.nn import GCNConv  
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn import svm
from sklearn.linear_model import LogisticRegression  


def predf(df):
    grouped = df.groupby('msisdn') 
    count_dst=grouped['call_event'].apply(lambda x:(x=='call_dst').sum())
    count_src=grouped['call_event'].apply(lambda x:(x=='call_src').sum())

    call_rate=count_dst/count_src
    call_rate.replace([np.inf, -np.inf], 100, inplace=True)
    user_stats = grouped['call_duration'].agg(['count', 'sum', 'mean'])

    call_type=grouped['phone1_type'].first()

    merged_stats = pd.merge(count_dst, count_src, on='msisdn')
    merged_stats = pd.merge(merged_stats,call_rate,on='msisdn')
    merged_stats = pd.merge(merged_stats,call_type,on='msisdn')
    merged_stats = pd.merge(merged_stats,user_stats,on='msisdn').reset_index()
    merged_stats.columns = ['msisdn','count_dst','count_src','call_rate','phone1_type','total_calls', 'total_duration', 'avg_call_duration']
    return merged_stats
if __name__ == "__main__":  
    # # 读取CSV文件  
    df1 = pd.read_csv('./trainSet_res.csv')  
    df_ans=pd.read_csv('./trainSet_ans.csv')  
    df2 = pd.read_csv('./validationSet_res.csv')  
    result=predf(df1)
    # print(result)
    merged_df=pd.merge(result, df_ans, on='msisdn', how='left')
    # 训练
    X_train=merged_df[['count_dst','count_src','call_rate','total_calls','phone1_type', 'total_duration', 'avg_call_duration']]
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    Y_train=merged_df['is_sa']

    model = svm.SVC(kernel='rbf', gamma='auto')  
    # model = LogisticRegression(max_iter=2)  

    model.fit(X_train_scaled, Y_train)

    #需要预测的
    result2=predf(df2)
    X_test=result2[['count_dst','count_src','call_rate','total_calls','phone1_type','total_duration', 'avg_call_duration']]
    X_test_scaled = scaler.fit_transform(X_test)
    y_pred = model.predict(X_test_scaled)

    test_result=pd.DataFrame({ 
            'msisdn': result2['msisdn'],
            'is_sa': y_pred
        })  
    test_result.to_csv('resultsvm2.csv', index=False)  # index=False 表示不保存索引 
    # print(test_result) 











    
    # 合并CSV文件（按行合并）  
    # merged_df = pd.concat([df1, df2], ignore_index=True)  
    # 将合并后的数据写入新的CSV文件 
    # G.add    
    # for _, row in df.iterrows():  
    #     if row['call_event']=="call_dst":
    #         caller, callee = row['msisdn'], row['other_party']
    #     else:
    #         callee, caller = row['msisdn'], row['other_party']
    #     edge_index.append((unique_users.tolist().index(caller), unique_users.tolist().index(callee)))  
    # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转换为PyTorch张量并转置  





    # df_res = pd.read_csv('./trainSet_res.csv',low_memory=False)
    # grouped = df_res.groupby('msisdn') 
    # user_stats = grouped['call_duration'].agg(['count', 'sum', 'mean']).reset_index()
    # user_stats.columns = ['msisdn', 'total_calls', 'total_duration', 'avg_call_duration']
    # user_stats.to_csv('user_call_statistics.csv', index=False)
    #数据预处理 通话次数 通话总时长 平均通话时长
    # df_val = pd.read_csv('./validationSet_res.csv',low_memory=False)
    # grouped = df_val.groupby('msisdn') 
    # user_stats = grouped['call_duration'].agg(['count', 'sum', 'mean']).reset_index()
    # user_stats.columns = ['msisdn', 'total_calls', 'total_duration', 'avg_call_duration']
    # user_stats.to_csv('user_call_statistics_val.csv', index=False)

    #验证集数据处理

    # user_stats=pd.read_csv('./user_call_statistics.csv')
    # df_ans=pd.read_csv('./trainSet_ans.csv')
    # merged_df=pd.merge(user_stats, df_ans, on='msisdn', how='left')
    # merged_df.to_csv('merged_df.csv', index=False)
    # 数据合并


    pass

  
  
 
