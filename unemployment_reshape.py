import pandas as pd

# 처리할 원본 파일 경로
source_file = 'hire_data_file/성_연령별_실업률_20250903155105.csv'
# 저장할 파일 경로
output_file = 'preprocessing/hiring/unemployment_long.csv'

print(f"'{source_file}' 파일 가공을 시작합니다...")

try:
    # 1. 원본 파일 읽기 (cp949 인코딩 사용)
    df = pd.read_csv(source_file, encoding='utf-8')

    # 2. 필요한 데이터만 필터링
    # 실제 컬럼 이름을 보면 '연령계층별'에 '계'가 아니라 '15세 이상'을 사용해야 전체 실업률을 얻을 수 있습니다.
    df_filtered = df[(df['성별'] == '계') & (df['연령계층별'] == '15세 이상')].copy()

    if df_filtered.empty:
        # 만약 '15세 이상'이 없다면 '계'를 시도
        df_filtered = df[(df['성별'] == '계') & (df['연령계층별'] == '계')].copy()

    # 3. 'Wide' -> 'Long' 포맷으로 변환 (pd.melt 사용)
    # id_vars: 기준이 될 열 (고정될 열)
    # var_name: 시간 열의 새로운 이름
    # value_name: 데이터 열의 새로운 이름
    df_long = df_filtered.melt(
        id_vars=['성별', '연령계층별'],
        var_name='date',
        value_name='unemployment_rate'
    )

    # 4. 날짜 형식 정리 및 최종 컬럼 선택
    # '1999.06' -> datetime 객체로 변환
    df_long['date'] = pd.to_datetime(df_long['date'], format='%Y.%m')

    # 필요한 컬럼만 남김
    final_df = df_long[['date', 'unemployment_rate']]

    # 5. 새로운 파일로 저장
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n'✅ {output_file}' 파일 생성 완료!")
    print("--- 가공된 데이터 (상위 5줄) ---")
    print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except KeyError as e:
    print(f"오류: 파일에서 예상된 컬럼({e})을 찾을 수 없습니다. 원본 파일의 컬럼 이름을 확인해주세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")