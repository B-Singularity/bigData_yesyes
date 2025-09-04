import pandas as pd

# --- 설정 ---
source_file = '노동생산성지수_부가가치기준__20250904095634_분석(전분기_대비_증감,증감률).csv'
output_file = 'preprocessing/productivity_all_long.csv'

# 필터링할 산업 이름 (파일에 있는 '비농전산업'이 사실상 전산업을 의미)
TARGET_INDUSTRY = '비농전산업'

print(f"'{source_file}' 파일 가공을 시작합니다...")

try:
    # 1. 원본 파일 읽기 (첫 두 줄을 멀티 헤더로 지정)
    df = pd.read_csv(source_file, encoding='utf-8', header=[0, 1])

    # 2. 필요한 데이터만 필터링 (비농전산업)
    # 멀티 헤더 때문에 컬럼 이름이 튜플 ('산업별(1)', '산업별(1)')이 됩니다.
    df_filtered = df[df[('산업별(1)', '산업별(1)')] == TARGET_INDUSTRY].copy()

    # 3. 분석에 불필요한 산업분류 컬럼들 삭제
    # drop을 위해 컬럼 이름 리스트를 만듭니다.
    industry_cols_to_drop = [col for col in df.columns if '산업별' in col[0]]
    df_processed = df_filtered.drop(columns=industry_cols_to_drop)

    # 4. 'Wide' -> 'Long' 포맷으로 변환 (stack 사용)
    # stack(level=0)은 멀티 인덱스 컬럼의 첫 번째 레벨(날짜)을 행으로 변환합니다.
    df_stacked = df_processed.stack(level=0)

    # 5. 인덱스를 컬럼으로 변환하고, 날짜 형식 정리
    df_long = df_stacked.reset_index()
    # level_0은 기존 행 번호이므로 삭제, level_1이 날짜 정보가 됩니다.
    df_long.rename(columns={'level_1': 'date'}, inplace=True)
    df_long.drop(columns=['level_0'], inplace=True)


    def convert_quarter_to_date(q_str):
        year, quarter_info = q_str.split('.')
        quarter = quarter_info.split('/')[0]
        # 분기 첫 달 1일로 변환 (월별 통일을 위해)
        month = (int(quarter) - 1) * 3 + 1
        return pd.to_datetime(f'{year}-{month}-01')


    df_long['date'] = df_long['date'].apply(convert_quarter_to_date)

    # 6. 최종 컬럼 이름 변경 및 저장
    final_df = df_long.rename(columns={
        '원데이터': 'productivity_index',
        '전분기 대비 증감': 'change_qoq',
        '증감률': 'growth_rate_qoq'
    })

    # 숫자 변환
    for col in ['productivity_index', 'change_qoq', 'growth_rate_qoq']:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n'✅ {output_file}' 파일 생성 완료!")
    print("--- 가공된 데이터 (상위 5줄) ---")
    print(final_df.head())

except Exception as e:
    print(f"오류가 발생했습니다: {e}")