import pandas as pd

df = pd.read_csv("../../investing_data_file/비목별_연구개발비_20250908005956.csv", header=[0, 1])

df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}" if x[1] else x[0])

software_df = df[df['비목별(2)_비목별(2)'] == '컴퓨터 소프트웨어']

columns_to_keep = ['비목별(1)_비목별(1)', '비목별(2)_비목별(2)'] + [col for col in df.columns if '연구개발비 (백만원)' in col]
software_df = software_df[columns_to_keep]

long_df = pd.melt(
    software_df,
    id_vars=['비목별(1)_비목별(1)', '비목별(2)_비목별(2)'],
    var_name='기간',
    value_name='금액'
)

long_df['기간'] = long_df['기간'].str.split('_').str[0]

long_df.rename(columns={'비목별(1)_비목별(1)': '비목별(1)', '비목별(2)_비목별(2)': '비목별(2)'}, inplace=True)

long_df.to_csv("software_long.csv", index=False)

print(long_df.head())