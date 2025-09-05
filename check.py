import pandas as pd
import os

source_file = "hire_data_file/기업경영판단_BSI_20250905140637.csv"

try:
    # header=0 옵션으로 CSV 파일을 정확히 읽습니다.
    df = pd.read_csv(source_file, header=0)

    # 첫 번째 컬럼 이름을 'Category'로 변경합니다.
    df.rename(columns={'기업경영판단별(1)': 'Category'}, inplace=True)

    # 따옴표와 공백을 모두 제거하여 정확한 카테고리 이름만 추출합니다.
    df['Category'] = df['Category'].str.replace('"', '').str.strip()

    # 중복을 제외한 모든 카테고리 이름을 찾습니다.
    unique_categories = df['Category'].unique()

    # 찾은 카테고리 목록을 출력합니다.
    print(f"'{source_file}' 파일에서 찾은 카테고리 전체 목록입니다:")
    print("----------------------------------------")
    for cat in unique_categories:
        print(f"- {cat}")
    print("----------------------------------------")

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"파일을 분석하는 중 오류가 발생했습니다: {e}")