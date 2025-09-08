import pandas as pd
import os

source_file = "../../investing_data_file/수출_및_수입액_20250907230946.csv"
output_folder = "outputs"
output_file = os.path.join(output_folder, "ict_export_import_total_long.csv")

try:
    # 1. 원본 CSV 파일을 2줄 헤더로 불러오기
    df = pd.read_csv(source_file, header=[0, 1])

    # 2. 복잡한 다중 헤더를 단일 헤더로 재구성
    new_columns = []
    id_vars_for_melt = []

    for col in df.columns:
        # 헤더의 첫 번째 부분이 날짜(숫자) 형태인지 확인하여 컬럼 구분
        try:
            float(str(col[0]).replace(' p)', '').strip())
            is_data_col = True
        except ValueError:
            is_data_col = False

        if is_data_col:
            # 데이터 컬럼이면 '연도_값종류' 형태로 이름을 합침 (예: '1999_수출액 (백만US$)')
            date_part = str(col[0]).strip()
            type_part = str(col[1]).strip()
            new_columns.append(f"{date_part}_{type_part}")
        else:
            # 설명 컬럼이면 아래쪽 헤더 이름을 사용
            col_name = str(col[1]).strip()
            new_columns.append(col_name)
            id_vars_for_melt.append(col_name)

    df.columns = new_columns

    # 3. (핵심 수정) 'ICT산업별(1)' 컬럼에서 '합계' 값만 필터링
    filtered_df = df[df['ICT산업별(1)'].str.strip() == '합계'].copy()

    # 4. Wide 포맷 -> Long 포맷으로 변환 (melt)
    long_df = filtered_df.melt(
        id_vars=id_vars_for_melt,
        var_name="Year_ValueType",
        value_name="금액(백만US$)"
    )

    # 5. 'Year_ValueType' 컬럼을 '연도'와 '구분'으로 분리
    long_df[['연도', '구분']] = long_df['Year_ValueType'].str.split('_', expand=True)

    # 6. 최종 데이터프레임 정리
    final_df = long_df.drop(columns=['Year_ValueType', 'ICT산업별(1)', 'ICT산업별(2)', 'ICT산업별(3)'])
    final_df['금액(백만US$)'] = pd.to_numeric(final_df['금액(백만US$)'], errors='coerce')
    final_df.dropna(subset=['금액(백만US$)'], inplace=True)
    final_df = final_df.sort_values(by=['연도', '구분']).reset_index(drop=True)

    # 7. CSV 파일로 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"합계 데이터 추출 및 변환 완료: {output_file}")
    print("\n### 최종 데이터 샘플 ###")
    print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")