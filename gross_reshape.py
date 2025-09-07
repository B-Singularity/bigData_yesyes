import pandas as pd

source_file = 'hire_data_file/경제활동별_총고정자본형성_실질__연간__20250907155231.csv'
output_file = 'preprocessing/hiring/GFCF_ICT_Real_long.csv'

try:
    # 2. CSV 파일을 읽어옵니다. (header=0 옵션으로 첫 줄을 제목으로 지정)
    df = pd.read_csv(source_file, header=0)

    # 3. 첫 번째 컬럼 이름을 'Category'로 바꾸고, 값을 깨끗하게 정리합니다.
    df.rename(columns={'계정항목별': 'Category'}, inplace=True)
    df['Category'] = df['Category'].str.replace('"', '').str.strip()

    # 4. '정보통신업' 데이터만 선택합니다.
    df_ict = df[df['Category'] == '정보통신업'].copy()

    if df_ict.empty:
        print("오류: 파일에서 '정보통신업' 카테고리를 찾을 수 없습니다.")
    else:
        # 5. Wide format을 Long format으로 변환합니다. (melt 함수 사용)
        df_long = df_ict.melt(id_vars=['Category'],
                              var_name='Year',
                              value_name='GFCF_Real') # GFCF: 총고정자본형성

        # 6. 최종 데이터를 정리합니다. (불필요한 Category 컬럼 제거)
        final_df = df_long[['Year', 'GFCF_Real']].sort_values(by='Year').reset_index(drop=True)
        final_df['Year'] = pd.to_numeric(final_df['Year'])

        # 7. 결과를 저장할 폴더를 만들고 CSV 파일로 저장합니다.
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"성공: '정보통신업' 데이터가 '{output_file}' 파일로 저장되었습니다.")
        print("\n변환된 데이터 샘플:")
        print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
except Exception as e:
    print(f"처리 중 오류가 발생했습니다: {e}")

