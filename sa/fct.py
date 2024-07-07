import pandas as pd
import numpy as np


def clean_and_aggregate_features(df, numeric_features, categorical_features):
    df_copy = df.copy()  # 显式创建 DataFrame 副本,好像必须有这个
    for cat_feat in categorical_features:
        df_copy.loc[:, cat_feat] = df_copy[cat_feat].astype(str)

    aggregation_functions = {feat: 'mean' for feat in numeric_features}
    aggregation_functions.update({feat: lambda x: x.mode()[0] if not x.mode().empty else 'unknown' for feat in categorical_features})
    aggregated_df = df_copy.groupby('msisdn').agg(aggregation_functions).reset_index()
    return aggregated_df

def convert_to_numeric(df,column):
    # 使用 `pd.to_numeric` 并设置 `errors='coerce'` 将无法转换的值替换为 NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')