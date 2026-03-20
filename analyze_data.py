import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel(r'C:\Users\32767\Desktop\3.20\main\咖啡店会员消费数据.xlsx')

print('=== 数据概览 ===')
print('数据形状:', df.shape)
print('\n列名:', df.columns.tolist())
print('\n前5行数据:')
print(df.head())
print('\n数据类型:')
print(df.dtypes)
print('\n数据基本统计:')
print(df.describe())
print('\n缺失值检查:')
print(df.isnull().sum())
