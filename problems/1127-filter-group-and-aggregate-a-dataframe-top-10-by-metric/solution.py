import pandas as pd

def solution(df):
    df = df[df['status']=='completed']
    df = df.groupby(by='region')['amount'].sum().reset_index()
    return df.sort_values('amount',ascending=False).head(10)