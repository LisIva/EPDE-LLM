import pandas as pd

file_path = 'kdv_sindy_metrics.csv'
df = pd.read_csv(file_path)
mean1 = df['runtime'].mean()
print()