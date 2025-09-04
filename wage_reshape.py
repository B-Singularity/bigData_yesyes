import pandas as pd
import re


def preprocess_type1(df, industry_col, scale_col):
    df.columns = df.columns.str.strip()

    filtered_rows = df[df[industry_col].str.contains('전산업', na=False) & df[scale_col].str.contains('전규모', na=False)]
    if filtered_rows.empty:
        return pd.DataFrame()

    target_row = filtered_rows.iloc[0]
    date_cols = [col for col in df.columns if re.match(r'\d{4}\.\d{2}\.\d{2}', col)]

    monthly_wages = []
    for i in range(0, len(date_cols), 3):
        year_month_cols = date_cols[i:i + 3]
        if len(year_month_cols) < 3:
            continue

        try:
            total_wage = pd.to_numeric(target_row[year_month_cols], errors='coerce').sum()
            year, month, _ = year_month_cols[0].split('.')
            monthly_wages.append({'Year': int(year), 'Month': int(month), 'Total_Wage': total_wage})
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(monthly_wages)

def preprocess_type2(df):
    filtered_df = df[
        df['항목'].str.contains('전체임금총액', na=False) & df['산업분류별'].str.contains('전체', na=False) & df['규모별'].str.contains(
            '전규모', na=False)]
    if filtered_df.empty:
        return pd.DataFrame()

    id_vars = ['산업분류별', '규모별', '항목', '단위']
    date_cols = [col for col in df.columns if '월' in str(col)]

    melted_df = filtered_df.melt(id_vars=id_vars, value_vars=date_cols, var_name='Date', value_name='Total_Wage')

    melted_df[['Year', 'Month']] = melted_df['Date'].str.extract(r'(\d{4})\.(\d{1,2})').astype(int)

    unit = melted_df['단위'].iloc[0]
    if unit == '천원':
        melted_df['Total_Wage'] = pd.to_numeric(melted_df['Total_Wage'], errors='coerce') * 1000
    else:
        melted_df['Total_Wage'] = pd.to_numeric(melted_df['Total_Wage'], errors='coerce')

    return melted_df[['Year', 'Month', 'Total_Wage']].dropna()


df1 = pd.read_csv('hire_data_file/99-01 임금.csv', encoding='utf-8', skiprows=1)
df2 = pd.read_csv('hire_data_file/93-98 임금.csv', encoding='utf-8', skiprows=1)
df3 = pd.read_csv('hire_data_file/19 임금.csv', encoding='cp949')
df4 = pd.read_csv('hire_data_file/15-18 임금.csv', encoding='cp949')
df5 = pd.read_csv('hire_data_file/11-14 임금.csv', encoding='cp949')
df6 = pd.read_csv('hire_data_file/08-10 임금.csv', encoding='cp949')
df7 = pd.read_csv('hire_data_file/02-07 임금.csv', encoding='utf-8', skiprows=1)

wage1 = preprocess_type1(df1, '산업별', '규모별')
wage2 = preprocess_type1(df2, '산업별', '규모별')
wage7 = preprocess_type1(df7, '산업별', '규모별')

wage3 = preprocess_type2(df3)
wage4 = preprocess_type2(df4)
wage5 = preprocess_type2(df5)
wage6 = preprocess_type2(df6)

all_wages = pd.concat([wage1, wage2, wage3, wage4, wage5, wage6, wage7], ignore_index=True)
all_wages_sorted = all_wages.sort_values(by=['Year', 'Month']).reset_index(drop=True)
all_wages_sorted.to_csv('월별_임금_총액(1993-2019).csv', index=False, encoding='utf-8-sig')

print(all_wages_sorted)
