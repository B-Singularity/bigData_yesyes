import pandas as pd
import os

# --- 설정 ---
source_file = "../../investing_data_file/설비투자지수_20250908010322.csv"
# 최종 결과물을 저장할 폴더와 파일 이름
output_folder = "outputs"
output_file = os.path.join(output_folder, "investment_final_data_all_categories.csv")

try:
    # 1. 원본 CSV 파일을 2줄 헤더로 불러오기
    df = pd.read_csv(source_file, header=[0, 1])

    # 2. 복잡한 다중 헤더를 단일 헤더로 재구성
    new_columns = []
    id_vars_for_melt = []
    for col in df.columns:
        try:
            float(str(col[0]).replace(' p)', '').strip())
            is_data_col = True
        except ValueError:
            is_data_col = False

        if is_data_col:
            date_part = str(col[0]).replace(' p)', '').strip()
            type_part = str(col[1]).strip()
            new_columns.append(f"{date_part}_{type_part}")
        else:
            col_name = str(col[1]).strip()
            new_columns.append(col_name)
            id_vars_for_melt.append(col_name)

    df.columns = new_columns

    # 3. Wide 포맷 -> Long 포맷으로 변환
    long_df = df.melt(
        id_vars=id_vars_for_melt,
        var_name="Date_ValueType",
        value_name="돈"
    )

    # 4. 'Date_ValueType' 컬럼을 '날짜'와 '지수종류'로 분리
    long_df[['날짜', '지수종류']] = long_df['Date_ValueType'].str.split('_', expand=True)
    long_df = long_df.drop(columns=['Date_ValueType'])

    # 5. (핵심 수정) 각기 다른 컬럼에 있는 모든 항목을 필터링
    cond1 = long_df['부문별(2020=100)(5)'].str.strip() == "반도체 제조용 기계"
    cond2 = long_df['부문별(2020=100)(4)'].str.strip() == "전기기기 및 장치"
    cond3 = long_df['부문별(2020=100)(4)'].str.strip() == "컴퓨터사무용기계"

    # 3가지 조건을 모두 만족하는 행들을 추출
    filtered_df = long_df[cond1 | cond2 | cond3].copy()

    # 6. '도메인' 컬럼 생성
    #    - 필터링된 각 항목의 이름을 하나의 '도메인' 컬럼으로 합침
    #    - cond2, cond3가 같은 컬럼을 공유하므로 먼저 처리
    filtered_df['도메인'] = filtered_df['부문별(2020=100)(4)']
    # cond1에 해당하는 '반도체 제조용 기계' 값으로 채워넣기
    filtered_df['도메인'].fillna(filtered_df['부문별(2020=100)(5)'], inplace=True)

    # 7. 최종 컬럼 선택
    final_df = filtered_df[['도메인', '날짜', '지수종류', '돈']]

    # 8. 최종 데이터 정리
    final_df['돈'] = pd.to_numeric(final_df['돈'], errors='coerce')
    final_df['날짜'] = pd.to_datetime(final_df['날짜'], format='%Y.%m')
    final_df.dropna(subset=['돈'], inplace=True)
    final_df = final_df.sort_values(by=['도메인', '날짜']).reset_index(drop=True)

    # 9. CSV 파일로 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"모든 카테고리를 포함한 최종 데이터 저장 완료: {output_file}")
    print("\n### 최종 추출된 데이터 샘플 (3개 카테고리 확인) ###")
    print("--- 컴퓨터사무용기계 ---")
    print(final_df[final_df['도메인'] == '컴퓨터사무용기계'].head(2))
    print("\n--- 반도체 제조용 기계 ---")
    print(final_df[final_df['도메인'] == '반도체 제조용 기계'].head(2))
    print("\n--- 전기기기 및 장치 ---")
    print(final_df[final_df['도메인'] == '전기기기 및 장치'].head(2))


except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")