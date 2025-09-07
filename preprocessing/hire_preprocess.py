import pandas as pd
import numpy as np
import os


def preprocess_bsi(source_file):
    """BSI 데이터를 전처리하여 종합 지수를 생성합니다."""
    print("1. BSI 데이터 전처리를 시작합니다...")
    df = pd.read_csv(source_file, header=0)
    df.rename(columns={'기업경영판단별(1)': 'Category'}, inplace=True)
    df['Category'] = df['Category'].str.replace('"', '').str.strip()
    df_long = df.melt(id_vars=['Category'], var_name='Date_Raw', value_name='Value')
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    df_long['Date_Raw'] = df_long['Date_Raw'].astype(str).apply(lambda x: '.'.join(x.split('.')[:2]))
    df_long['Type_Num'] = df_long.groupby(['Category', 'Date_Raw']).cumcount()
    type_map = {0: '전망', 1: '실적'}
    df_long['Type'] = df_long['Type_Num'].map(type_map)
    df_long['Date'] = pd.to_datetime(df_long['Date_Raw'], format='%Y.%m')
    df_actuals = df_long[df_long['Type'] == '실적'].copy()
    df_pivot = df_actuals.pivot_table(index='Date', columns='Category', values='Value')
    target_categories = ['생산증가율', '내수판매', '수출']
    existing_cols = [col for col in target_categories if col in df_pivot.columns]
    if not existing_cols:
        raise ValueError(f"BSI 파일에서 종합 지수 계산에 필요한 카테고리를 찾지 못했습니다.")
    df_pivot['BSI_Composite'] = df_pivot[existing_cols].mean(axis=1)
    final_df_bsi = df_pivot[['BSI_Composite']].reset_index()
    print("-> 성공: BSI 종합 지수 생성 완료.")
    return final_df_bsi


def preprocess_gfcf_ict(source_file):
    """총고정자본형성 데이터에서 '정보통신업'만 추출하고 전처리합니다."""
    print("\n2. 총고정자본형성(GFCF) 데이터 전처리를 시작합니다...")
    df = pd.read_csv(source_file, header=0)
    df.rename(columns={'계정항목별': 'Category'}, inplace=True)
    df['Category'] = df['Category'].str.replace('"', '').str.strip()
    df_ict = df[df['Category'] == '정보통신업'].copy()
    if df_ict.empty:
        raise ValueError("GFCF 파일에서 '정보통신업' 카테고리를 찾을 수 없습니다.")
    df_long = df_ict.melt(id_vars=['Category'], var_name='Year', value_name='GFCF_ICT_Real')
    final_df = df_long[['Year', 'GFCF_ICT_Real']].sort_values(by='Year').reset_index(drop=True)
    final_df['Year'] = pd.to_numeric(final_df['Year'])
    print("-> 성공: GFCF 정보통신업 데이터 추출 완료.")
    return final_df


def align_and_feature_engineer(base_file, cpi_file, df_bsi, df_gfcf):
    """모든 데이터를 통합하고 기본 및 고급 특성 공학을 적용합니다."""
    print("\n3. 모든 데이터 통합 및 정렬을 시작합니다...")
    df_base = pd.read_csv(base_file, header=0)
    df_base.rename(columns={df_base.columns[0]: 'Date'}, inplace=True)
    df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base['Year'] = df_base['Date'].dt.year

    df_cpi_raw = pd.read_csv(cpi_file, header=0)
    df_cpi_raw.rename(columns={'시도별': 'Category'}, inplace=True)
    df_cpi_raw['Category'] = df_cpi_raw['Category'].str.replace('"', '').str.strip()
    df_cpi_raw.set_index('Category', inplace=True)
    cpi_nationwide = df_cpi_raw.loc['전국'].iloc[1::2]
    cpi_nationwide.index = pd.to_datetime(cpi_nationwide.index.map(lambda x: '.'.join(str(x).split('.')[:2])),
                                          format='%Y.%m')
    df_cpi = cpi_nationwide.reset_index()
    df_cpi.columns = ['Date', 'cpi_growth']
    df_cpi['cpi_growth'] = pd.to_numeric(df_cpi['cpi_growth'], errors='coerce')

    merged_df1 = pd.merge(df_base, df_gfcf, on='Year', how='left')
    merged_df2 = pd.merge(merged_df1, df_bsi, on='Date', how='left')
    final_df = pd.merge(merged_df2, df_cpi, on='Date', how='left')

    final_df['wage_growth'] = final_df['total_wage'].pct_change() * 100
    final_df['real_wage_growth'] = final_df['wage_growth'] - final_df['cpi_growth']
    print("-> 성공: 데이터 통합 및 기본 피처 생성 완료.")

    print("\n4. 기본 특성 공학을 시작합니다...")
    final_df['unemployment_rate_MA3'] = final_df['unemployment_rate'].rolling(window=3).mean()
    final_df['BSI_Composite_MA3'] = final_df['BSI_Composite'].rolling(window=3).mean()
    final_df['unemployment_rate_change3'] = final_df['unemployment_rate'].diff(3)
    final_df['BSI_Composite_change3'] = final_df['BSI_Composite'].diff(3)
    print("-> 성공: 단기 추세 변수 추가 완료.")

    # --- 여기가 핵심 추가 부분입니다 (고급 특성 공학) ---
    print("\n5. 고급 특성 공학을 시작합니다...")
    # (5a) 중장기 추세 변수 추가 (6개월, 12개월)
    final_df['unemployment_rate_MA6'] = final_df['unemployment_rate'].rolling(window=6).mean()
    final_df['BSI_Composite_MA12'] = final_df['BSI_Composite'].rolling(window=12).mean()
    final_df['real_wage_growth_change12'] = final_df['real_wage_growth'].diff(12)

    # (5b) 변동성 변수 추가 (3개월, 6개월 표준편차)
    final_df['unemployment_rate_std3'] = final_df['unemployment_rate'].rolling(window=3).std()
    final_df['BSI_Composite_std6'] = final_df['BSI_Composite'].rolling(window=6).std()
    print("-> 성공: 중장기 추세 및 변동성 변수 추가 완료.")

    final_df['growth_rate_qoq_lag1'] = final_df['growth_rate_qoq'].shift(1)
    print("-> 성공: lag1 변수 추가 완료.")

    return final_df


if __name__ == '__main__':
    try:
        # --- 설정 ---
        BSI_SOURCE_FILE = '../hire_data_file/기업경영판단_BSI_20250905140637.csv'
        BASE_DATA_FILE = 'aligned_data_year_month.csv'
        CPI_SOURCE_FILE = '../hire_data_file/소비자물가지수_2020100__20250905135045_분석(전월_대비_증감률).csv'
        GFCF_SOURCE_FILE = '../hire_data_file/경제활동별_총고정자본형성_실질__연간__20250907155231.csv'
        OUTPUT_FOLDER = 'preprocessing'
        FINAL_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'final_analytical_data_advanced_featured.csv')

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        # --- 실행 ---
        preprocessed_bsi_df = preprocess_bsi(BSI_SOURCE_FILE)
        preprocessed_gfcf_df = preprocess_gfcf_ict(GFCF_SOURCE_FILE)

        final_featured_df = align_and_feature_engineer(
            BASE_DATA_FILE, CPI_SOURCE_FILE, preprocessed_bsi_df, preprocessed_gfcf_df
        )

        final_featured_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("\n--- 모든 전처리 작업 완료 ---")
        print(f"고급 특성이 추가된 최종 분석용 데이터가 '{FINAL_OUTPUT_FILE}'에 저장되었습니다.")
        print("생성된 최종 데이터 샘플 (마지막 컬럼들 확인):")
        print(final_featured_df.tail())

    except FileNotFoundError as e:
        print(f"\n[오류] 파일 없음: '{e.filename}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")