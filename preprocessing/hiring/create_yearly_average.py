import pandas as pd
import os

input_file = 'final_analytical_data_featured.csv'
output_file = 'yearly_average.csv'

try:
    df_monthly = pd.read_csv(input_file)
    print(f"'{input_file}' 파일을 성공적으로 불러왔습니다.")

    df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
    df_monthly['Year'] = df_monthly['Date'].dt.year

    df_yearly_average = df_monthly.groupby('Year').mean(numeric_only=True)

    df_yearly_average = df_yearly_average.reset_index()

    df_yearly_average.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(df_yearly_average.head())

except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")




