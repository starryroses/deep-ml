import pandas as pd

def solution(df1, df2, df3):
    merged = pd.merge(df1,df2,how='inner',on='emp_id')
    merged_df3 = pd.merge(merged,df3,how='left',on='emp_id')
    return merged_df3.reset_index(drop=True)