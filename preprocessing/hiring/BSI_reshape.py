import pandas as pd
import numpy as np
import os

# --- 설정 ---
source_file = "../../hire_data_file/기업경영판단_BSI_20250905140637.csv"
output_file = "BSI_long.csv"

# --- 전처리 실행 ---
try:
    # CSV 파일 읽기
    print("=== CSV 파일 구조 분석 ===")
    df = pd.read_csv(source_file, header=0)
    print(f"데이터 형태: {df.shape}")
    print(f"컬럼명 수: {len(df.columns)}")

    # 첫 번째 컬럼을 Category로 변경
    df.rename(columns={df.columns[0]: 'Category'}, inplace=True)
    df['Category'] = df['Category'].astype(str).str.replace('"', '').str.strip()

    print(f"Category: {df['Category'].tolist()}")

    # 데이터의 실제 값들을 확인해보기
    print(f"\n=== 데이터 값 샘플 확인 ===")
    for i, col in enumerate(df.columns[1:6]):  # 처음 5개 컬럼만 확인
        print(f"컬럼 '{col}'의 unique 값들: {df[col].unique()}")

    # 날짜 컬럼들을 실적/전망으로 분류
    date_cols = [col for col in df.columns[1:] if col != 'Category']

    # 실적 컬럼: .1이 없는 것들
    actual_cols = [col for col in date_cols if not col.endswith('.1')]
    # 전망 컬럼: .1이 있는 것들
    forecast_cols = [col for col in date_cols if col.endswith('.1')]

    print(f"\n실적 컬럼 수: {len(actual_cols)}")
    print(f"전망 컬럼 수: {len(forecast_cols)}")
    print(f"실적 컬럼 예시: {actual_cols[:5]}")
    print(f"전망 컬럼 예시: {forecast_cols[:5]}")

    # 데이터를 숫자로 변환 (에러가 있는 값들은 NaN으로)
    df_clean = df.copy()

    # 실적 데이터만 처리
    print(f"\n=== 실적 데이터 처리 ===")
    for col in actual_cols:
        # 숫자로 변환 시도
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 실적 데이터만 추출하여 long format으로 변환
    actual_data = df_clean[['Category'] + actual_cols].copy()

    print(f"실적 데이터 형태: {actual_data.shape}")
    print("실적 데이터 샘플:")
    print(actual_data.head())

    # Long format으로 변환
    df_long = actual_data.melt(id_vars=['Category'],
                               var_name='Date_Raw',
                               value_name='BSI_Value')

    # 날짜 변환
    df_long['Date'] = pd.to_datetime(df_long['Date_Raw'], format='%Y.%m', errors='coerce')

    # NaN 값 제거
    df_long = df_long.dropna(subset=['BSI_Value', 'Date'])

    print(f"\nLong format 데이터 형태: {df_long.shape}")
    print("Long format 샘플:")
    print(df_long.head())

    # 종합 지수를 만들기 위한 대상 항목들
    target_categories = ['생산증가율', '내수판매', '수출']

    available_categories = df_long['Category'].unique().tolist()
    existing_categories = [cat for cat in target_categories if cat in available_categories]

    print(f"\n사용 가능한 카테고리: {available_categories}")
    print(f"목표 카테고리 중 존재하는 것: {existing_categories}")

    if not existing_categories:
        print(f"경고: 목표 카테고리 {target_categories}와 일치하는 것이 없습니다.")
        print("대신 모든 카테고리의 평균을 사용합니다.")
        existing_categories = available_categories

    # 대상 카테고리의 데이터만 필터링
    target_data = df_long[df_long['Category'].isin(existing_categories)]

    print(f"\n목표 데이터 형태: {target_data.shape}")

    if len(target_data) > 0:
        # 각 날짜별로 종합 지수 계산
        composite_by_date = target_data.groupby('Date')['BSI_Value'].mean().reset_index()
        composite_by_date.rename(columns={'BSI_Value': 'BSI_Composite'}, inplace=True)

        print(f"\n종합 지수 계산 완료:")
        print(f"사용된 카테고리: {existing_categories}")
        print(f"날짜 범위: {composite_by_date['Date'].min()} ~ {composite_by_date['Date'].max()}")
        print(f"데이터 포인트 수: {len(composite_by_date)}")

        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 파일 저장
        composite_by_date.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n성공: '{output_file}' 파일이 생성되었습니다.")
        print("\n최종 결과 (처음 10행):")
        print(composite_by_date.head(10))
        print("\n최종 결과 (마지막 10행):")
        print(composite_by_date.tail(10))

        # 통계 정보
        print(f"\nBSI 종합 지수 통계:")
        print(f"평균: {composite_by_date['BSI_Composite'].mean():.2f}")
        print(f"최솟값: {composite_by_date['BSI_Composite'].min():.2f}")
        print(f"최댓값: {composite_by_date['BSI_Composite'].max():.2f}")
        print(f"표준편차: {composite_by_date['BSI_Composite'].std():.2f}")

    else:
        print("오류: 처리할 수 있는 데이터가 없습니다.")

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")
    import traceback

    traceback.print_exc()