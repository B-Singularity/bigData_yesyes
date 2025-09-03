import pandas as pd

# --- 설정 ---
source_file = '성_및_연령별_추계인구_1세별__5세별____전국_20250903135125.csv'
output_file = 'population_long.csv'

# 더해야 할 '생산가능인구' 연령대 목록 (파일에 있는 이름과 정확히 일치해야 함)
TARGET_AGE_GROUPS = [
    '15 - 19세', '20 - 24세', '25 - 29세', '30 - 34세', '35 - 39세',
    '40 - 44세', '45 - 49세', '50 - 54세', '55 - 59세', '60 - 64세'
]

print(f"'{source_file}' 파일 가공을 시작합니다...")

try:
    # 1. 원본 파일 읽기
    df = pd.read_csv(source_file, encoding='utf-8')

    # 2. 필요한 데이터만 필터링 (전체 성별, 생산가능인구 연령대)
    df_filtered = df[
        (df['성별'] == '전체') &
        (df['연령별'].isin(TARGET_AGE_GROUPS))
        ].copy()

    # 3. 연도 컬럼들만 선택하여 합산 준비
    # (가정별, 성별, 연령별 컬럼을 제외한 모든 컬럼을 연도 컬럼으로 간주)
    year_columns = df.columns.drop(['가정별', '성별', '연령별'])

    # '-'와 같은 문자열을 숫자로 변환 (오류 발생 시 0으로 처리)
    df_filtered[year_columns] = df_filtered[year_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 4. 연령대별 인구수를 연도별로 모두 합산하여 하나의 행으로 만듦
    df_sum = df_filtered[year_columns].sum().to_frame().T

    # 5. 합산된 데이터를 'Long' 포맷으로 변환 (pd.melt)
    df_long = df_sum.melt(
        var_name='date',
        value_name='population'
    )

    # 6. 날짜 형식 정리 및 최종 컬럼 선택
    df_long['date'] = pd.to_datetime(df_long['date'], format='%Y') + pd.offsets.YearEnd(0)
    final_df = df_long[['date', 'population']]

    # 7. 새로운 파일로 저장
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n'✅ {output_file}' 파일 생성 완료!")
    print("--- 가공된 데이터 (상위 5줄) ---")
    print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except KeyError as e:
    print(f"오류: 파일에서 예상된 컬럼({e})을 찾을 수 없습니다. 원본 파일의 헤더를 확인해주세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")