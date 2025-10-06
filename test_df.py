import pandas as pd
df = pd.read_parquet('eval_results/epoch_1d/dataframes/iterative_gpt-5-2025-08-07.parquet')

# 1. 按索引打印
print(df.iloc[5])  # 打印第一行
# print(df.iloc[5])

# 2. 按条件筛选后打印
# print(df[df['qid'] == 1].iloc[0])  # QID=1的第一行
# print(df[(df['task'] == 'cfl') & (df['precision_level'] == 'high')].iloc[0])

# 3. 美化打印
# for col, val in df.iloc[0].items():
#     print(f'{col}: {val}')
