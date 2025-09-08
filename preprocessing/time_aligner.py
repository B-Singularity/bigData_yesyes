import pandas as pd
import numpy as np
import os
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA


class TimeAligner:
    def __init__(self, start_date: str = '1995-01', end_date: str = '2025-12', freq: str = 'MS'):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.master_index = pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        self.datasets = {}

    def add_preprocessed_df(self, df: pd.DataFrame, dataset_name: str, date_col: str, data_col: str):
        if df is None or df.empty:
            print(f"경고: '{dataset_name}'에 대한 데이터프레임이 비어있어 건너뜁니다.")
            return
        if date_col not in df.columns or data_col not in df.columns:
            print(f"경고: '{dataset_name}' 처리 중 필요한 컬럼('{date_col}' 또는 '{data_col}')을 찾을 수 없어 건너뜁니다.")
            return

        df.columns = df.columns.str.strip()
        temp_df = df[[date_col, data_col]].copy()
        temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df['date'] = temp_df['date'].dt.to_period('M').dt.to_timestamp()

        temp_df.set_index('date', inplace=True)
        if not temp_df.index.is_unique:
            temp_df = temp_df.groupby('date').mean()

        self.datasets[dataset_name] = temp_df
        print(f"-> '{dataset_name}' 데이터가 처리 목록에 추가되었습니다.")

    def add_yearly_df_as_monthly(self, df: pd.DataFrame, dataset_name: str, date_col: str, data_col: str):
        if df is None or df.empty:
            print(f"경고: '{dataset_name}'에 대한 데이터프레임이 비어있어 건너뜁니다.")
            return
        if date_col not in df.columns or data_col not in df.columns:
            print(f"경고: '{dataset_name}' 처리 중 필요한 컬럼('{date_col}' 또는 '{data_col}')을 찾을 수 없어 건너뜁니다.")
            return

        temp_df = df[[date_col, data_col]].copy()
        temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
        temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce').dt.to_period('A').dt.to_timestamp()
        temp_df.dropna(subset=['date'], inplace=True)

        temp_df.set_index('date', inplace=True)
        if not temp_df.index.is_unique:
            temp_df = temp_df.groupby('date').mean()

        df_monthly = temp_df.resample('MS').asfreq().interpolate(method='linear')
        df_monthly[dataset_name] = df_monthly[dataset_name] / 12

        self.datasets[dataset_name] = df_monthly
        print(f"-> '{dataset_name}' 연간 데이터가 월간으로 변환되어 목록에 추가되었습니다.")

    def _backcast_with_arima(self, series_to_forecast: pd.Series) -> pd.Series:
        series_to_forecast = series_to_forecast.asfreq('MS').interpolate()
        print(f"\n'{series_to_forecast.name}'에 대한 AutoARIMA 최적 모델 탐색...")
        auto_model = AutoARIMA(season_length=12, trace=False, stepwise=True)
        sf = auto_model.fit(y=series_to_forecast.values)
        best_order = sf.model_.get('order', (1, 1, 1))
        best_seasonal_order = sf.model_.get('seasonal_order', (0, 0, 0, 0))
        model = ARIMA(series_to_forecast, order=best_order, seasonal_order=best_seasonal_order)
        fitted_model = model.fit()
        backcast_start = self.master_index[0]
        backcast_end = series_to_forecast.index[0] - pd.DateOffset(months=1)
        if backcast_start > backcast_end: return None
        print(f"'{series_to_forecast.name}' 후방예측 수행: {backcast_start.date()} ~ {backcast_end.date()}")
        return fitted_model.predict(start=backcast_start, end=backcast_end)

    def _fallback_impute_with_hybrid_model(self, master_df: pd.DataFrame, col: str, series: pd.Series):
        print(f"'{col}' ARIMA 예측 실패 또는 데이터 부족. 하이브리드 보간법(Plan B)을 적용합니다.")
        mean, std_dev = series.mean(), series.std()
        if pd.isnull(std_dev) or std_dev == 0: std_dev = 0
        missing_indices = master_df.index[master_df.index < series.index[0]]
        num_missing = len(missing_indices)
        if num_missing == 0: return

        avg_slope = series.iloc[:4].diff().mean()
        if pd.isnull(avg_slope): avg_slope = 0
        first_valid_val = series.iloc[0]
        extrapolated_values = first_valid_val - np.arange(num_missing, 0, -1) * avg_slope
        noise = np.random.normal(loc=0, scale=std_dev / 10, size=num_missing)
        master_df.loc[missing_indices, col] = extrapolated_values + noise

    def _merge_and_impute(self) -> pd.DataFrame:
        master_df = pd.DataFrame(index=self.master_index)
        for name, df in self.datasets.items():
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep='first')]
            master_df = master_df.join(df, how='outer')

        master_df = master_df.loc[self.start_date:self.end_date].resample(self.freq).mean()

        master_df.interpolate(method='linear', limit_direction='both', inplace=True)

        for col in master_df.columns:
            if pd.isnull(master_df[col].iloc[0]):
                series = master_df[col].dropna()
                if not series.empty and len(series) > 24:
                    try:
                        backcast = self._backcast_with_arima(series)
                        if backcast is not None: master_df[col].update(backcast)
                    except Exception as e:
                        print(f"ARIMA 오류 발생 ({col}): {e}")
                        self._fallback_impute_with_hybrid_model(master_df, col, series)
                elif not series.empty:
                    self._fallback_impute_with_hybrid_model(master_df, col, series)

        master_df.ffill(inplace=True)
        master_df.bfill(inplace=True)
        return master_df

    def process_and_save(self, output_path: str) -> pd.DataFrame:
        if not self.datasets:
            print("오류: 처리할 데이터셋이 없습니다.")
            return None
        print("\n--- 3. 데이터 병합 및 보간 시작 ---")
        final_data = self._merge_and_impute()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_data.to_csv(output_path)
        print(f"\n데이터를 '{output_path}' 경로에 CSV 파일로 저장했습니다.")
        return final_data


# 수정된 software 전처리 함수 - 더 상세한 디버깅
def preprocess_software(file_path):
    print(f"전처리: {os.path.basename(file_path)}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"CSV 파일 읽기 실패: {e}")
        return pd.DataFrame()

    # 디버깅: 원본 데이터 확인
    print(f"원본 데이터 shape: {df.shape}")
    print(f"컬럼명: {df.columns.tolist()}")

    if '비목별(2)' not in df.columns:
        print("경고: '비목별(2)' 컬럼이 없습니다.")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        # 소프트웨어 관련 컬럼을 찾아보기
        software_cols = [col for col in df.columns if '소프트웨어' in col or 'software' in col.lower()]
        if software_cols:
            print(f"소프트웨어 관련 컬럼: {software_cols}")
        return pd.DataFrame()

    print(f"비목별(2) 고유값: {df['비목별(2)'].unique()}")

    # 소프트웨어 관련 모든 항목 찾기
    software_keywords = ['소프트웨어', '프로그램', 'SW', 'software', '컴퓨터 소프트웨어']
    software_mask = df['비목별(2)'].str.contains('|'.join(software_keywords), case=False, na=False)
    df_software_all = df[software_mask].copy()

    print(f"소프트웨어 관련 항목들: {df_software_all['비목별(2)'].unique()}")
    print(f"소프트웨어 관련 데이터 shape: {df_software_all.shape}")

    if df_software_all.empty:
        print("소프트웨어 관련 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()

    # 가장 적합한 항목 선택 (컴퓨터 소프트웨어 우선, 없으면 가장 첫 번째)
    if '컴퓨터 소프트웨어' in df_software_all['비목별(2)'].values:
        df_filtered = df_software_all[df_software_all['비목별(2)'] == '컴퓨터 소프트웨어'].copy()
        print("'컴퓨터 소프트웨어' 데이터 사용")
    else:
        # 가장 데이터가 많은 소프트웨어 항목 선택
        software_counts = df_software_all['비목별(2)'].value_counts()
        best_software = software_counts.index[0]
        df_filtered = df_software_all[df_software_all['비목별(2)'] == best_software].copy()
        print(f"'{best_software}' 데이터 사용 (가장 많은 데이터)")

    print(f"최종 선택된 데이터 shape: {df_filtered.shape}")

    if df_filtered.empty:
        return pd.DataFrame()

    # 컬럼 확인
    if '금액' not in df_filtered.columns:
        print("'금액' 컬럼을 찾을 수 없습니다.")
        money_cols = [col for col in df_filtered.columns if '금액' in col or '액' in col or 'amount' in col.lower()]
        if money_cols:
            print(f"금액 관련 컬럼: {money_cols}")
            df_filtered.rename(columns={money_cols[0]: '금액'}, inplace=True)
        else:
            print("금액 관련 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()

    if '기간' not in df_filtered.columns:
        print("'기간' 컬럼을 찾을 수 없습니다.")
        date_cols = [col for col in df_filtered.columns if
                     '기간' in col or '날짜' in col or '연도' in col or 'date' in col.lower()]
        if date_cols:
            print(f"날짜 관련 컬럼: {date_cols}")
            df_filtered.rename(columns={date_cols[0]: '기간'}, inplace=True)
        else:
            print("날짜 관련 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()

    # 숫자 변환 및 null 처리
    print(f"금액 컬럼 변환 전 샘플: {df_filtered['금액'].head()}")
    print(f"금액 컬럼 데이터 타입: {df_filtered['금액'].dtype}")

    df_filtered['금액'] = pd.to_numeric(df_filtered['금액'], errors='coerce')

    # null 값 확인
    null_count_before = df_filtered['금액'].isnull().sum()
    total_count = len(df_filtered)
    print(f"숫자 변환 후 null 개수: {null_count_before}/{total_count}")

    if null_count_before == total_count:
        print("모든 금액 데이터가 null입니다. 원본 데이터 문제 가능성이 높습니다.")
        return pd.DataFrame()

    df_filtered.dropna(subset=['금액'], inplace=True)
    print(f"null 제거 후 shape: {df_filtered.shape}")

    # 기간 정보 확인 및 변환
    print(f"기간 컬럼 샘플: {df_filtered['기간'].head()}")
    print(f"기간 컬럼 데이터 타입: {df_filtered['기간'].dtype}")

    # 다양한 날짜 형식 시도
    date_formats = ['%Y', '%Y-%m', '%Y.%m', '%Y/%m', '%Y년', '%Y년%m월']
    converted = False

    for fmt in date_formats:
        try:
            df_filtered['기간'] = pd.to_datetime(df_filtered['기간'], format=fmt, errors='coerce')
            non_null_dates = df_filtered['기간'].dropna()
            if len(non_null_dates) > 0:
                print(f"날짜 변환 성공 (형식: {fmt}), 범위: {non_null_dates.min()} ~ {non_null_dates.max()}")
                converted = True
                break
        except Exception as e:
            continue

    if not converted:
        try:
            df_filtered['기간'] = pd.to_datetime(df_filtered['기간'], errors='coerce')
            non_null_dates = df_filtered['기간'].dropna()
            if len(non_null_dates) > 0:
                print(f"자동 날짜 변환 성공, 범위: {non_null_dates.min()} ~ {non_null_dates.max()}")
                converted = True
        except Exception as e:
            print(f"날짜 변환 실패: {e}")

    if not converted:
        print("날짜 변환에 실패했습니다.")
        return pd.DataFrame()

    # 날짜 null 제거
    df_filtered.dropna(subset=['기간'], inplace=True)
    print(f"날짜 null 제거 후 최종 shape: {df_filtered.shape}")

    if df_filtered.empty:
        print("최종 데이터가 비어있습니다.")
        return pd.DataFrame()

    # 중복 제거 및 집계
    if df_filtered.duplicated(['기간']).any():
        print("중복 기간 데이터 발견, 평균값으로 집계합니다.")
        df_filtered = df_filtered.groupby('기간')['금액'].mean().reset_index()

    print(f"최종 처리된 데이터: {len(df_filtered)}개 행")
    print(f"데이터 요약:\n{df_filtered.describe()}")

    return df_filtered


def preprocess_ict_export_import(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df_pivot = df.pivot_table(index='연도', columns='구분', values='금액(백만US$)').reset_index()
    df_pivot.columns = ['연도', 'ICT_Import', 'ICT_Export']
    return df_pivot


def preprocess_info_comm(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df['날짜'] = pd.to_datetime(df['기간'].str.split('/').str[0], format='%Y.%m')
    return df


def preprocess_investment(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df['지수종류'] = df['지수종류'].str.strip()
    df_pivot = df.pivot_table(index='날짜', columns=['도메인', '지수종류'], values='돈')
    df_pivot.columns = ['_'.join(col).strip().replace(' (2020=100)', '') for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)
    return df_pivot


def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"경고: 파일 없음 '{os.path.basename(file_path)}', 건너뜁니다.")
        return None


if __name__ == '__main__':
    DATA_DIRECTORY = './'
    file_paths = {
        "ict_export_import": os.path.join(DATA_DIRECTORY, 'investing/ict_export_import_total_long.csv'),
        "info_comm": os.path.join(DATA_DIRECTORY, 'investing/info_comm_long.csv'),
        "investment_raw": os.path.join(DATA_DIRECTORY, 'investing/investment_final_data_all_categories.csv'),
        "software": os.path.join(DATA_DIRECTORY, 'investing/software_long.csv'),
        "exchange_rate_raw": os.path.join(DATA_DIRECTORY, 'investing/korea_exchange_rate_long.csv'),
        "gfcf_total_raw": os.path.join(DATA_DIRECTORY, 'investing/GFCF_ICT_Real_long.csv'),
        "corporate_loans_raw": os.path.join(DATA_DIRECTORY, 'investing/corporate_loans_long.csv'),
        "corporate_loans_lagged": os.path.join(DATA_DIRECTORY, 'investing/corporate_loans_lagged.csv'),
        "gfcf_ma6": os.path.join(DATA_DIRECTORY, 'investing/GFCF_Real_ma6.csv'),
        "investment_lagged": os.path.join(DATA_DIRECTORY, 'investing/investment_lagged_data.csv'),
        "investment_ma": os.path.join(DATA_DIRECTORY, 'investing/investment_moving_averages.csv'),
        "exchange_rate_lagged": os.path.join(DATA_DIRECTORY, 'investing/korea_exchange_rate_lagged.csv'),
    }

    try:
        print("--- 1. 개별 데이터 전처리 시작 ---")
        df_ict_exp_imp = preprocess_ict_export_import(file_paths["ict_export_import"])
        df_info_comm = preprocess_info_comm(file_paths["info_comm"])
        df_investment = preprocess_investment(file_paths["investment_raw"])

        # Software 데이터 처리 - 디버깅 포함
        print("\n=== SOFTWARE 데이터 디버깅 ===")
        df_software = preprocess_software(file_paths["software"])

        print("\n--- 2. 데이터 정렬 및 통합 시작 ---")
        aligner = TimeAligner()

        aligner.add_yearly_df_as_monthly(df_ict_exp_imp, 'ICT_Export', '연도', 'ICT_Export')
        aligner.add_yearly_df_as_monthly(df_ict_exp_imp, 'ICT_Import', '연도', 'ICT_Import')
        aligner.add_yearly_df_as_monthly(safe_read_csv(file_paths["exchange_rate_raw"]), 'Exchange_Rate', '연도', '환율')
        aligner.add_yearly_df_as_monthly(safe_read_csv(file_paths["gfcf_total_raw"]), 'GFCF_Total_Real', 'Year',
                                         'GFCF_Real')

        # Software 데이터 추가 - 조건부 처리
        if not df_software.empty:
            print(f"Software 데이터 추가 시도: shape={df_software.shape}")
            aligner.add_yearly_df_as_monthly(df_software, 'Software_Investment', '기간', '금액')
        else:
            print("Software 데이터가 비어있어 건너뜁니다.")

        aligner.add_preprocessed_df(safe_read_csv(file_paths["corporate_loans_raw"]), 'Corporate_Loan_Rate', '날짜',
                                    '금리(연리%)')
        aligner.add_preprocessed_df(df_info_comm, 'GFCF_InfoComm_Quarterly', '날짜', '값')

        investment_cols = [col for col in df_investment.columns if col != '날짜']
        for col in investment_cols:
            aligner.add_preprocessed_df(df_investment, col, '날짜', col)

        # 나머지 데이터 처리는 동일...
        df_loans_lagged = safe_read_csv(file_paths["corporate_loans_lagged"])
        if df_loans_lagged is not None:
            aligner.add_preprocessed_df(df_loans_lagged, 'Corporate_Loan_Rate_lag_6', '날짜', '금리(연리%)_lag_6')

        df_gfcf_ma6 = safe_read_csv(file_paths["gfcf_ma6"])
        if df_gfcf_ma6 is not None:
            aligner.add_yearly_df_as_monthly(df_gfcf_ma6, 'GFCF_Total_Real_ma_6', 'Year', 'GFCF_Real_6년_이동평균')

        df_exchange_lagged = safe_read_csv(file_paths["exchange_rate_lagged"])
        if df_exchange_lagged is not None:
            for col in [c for c in df_exchange_lagged.columns if 'lag' in c]:
                aligner.add_yearly_df_as_monthly(df_exchange_lagged, f'Exchange_Rate_{col}', '연도', col)

        df_inv_lagged = safe_read_csv(file_paths["investment_lagged"])
        if df_inv_lagged is not None:
            df_inv_lagged['날짜'] = pd.to_datetime(df_inv_lagged['날짜'])
            lagged_cols_to_process = [c for c in df_inv_lagged.columns if 'lag' in c]
            for col in lagged_cols_to_process:
                temp_pivot = df_inv_lagged.pivot_table(index='날짜', columns='도메인', values=col).reset_index()
                temp_pivot.columns = [f"{c}_{col}" if c != '날짜' else c for c in temp_pivot.columns]
                for pivoted_col in [c for c in temp_pivot.columns if c != '날짜']:
                    aligner.add_preprocessed_df(temp_pivot, pivoted_col, '날짜', pivoted_col)

        df_inv_ma = safe_read_csv(file_paths["investment_ma"])
        if df_inv_ma is not None:
            df_inv_ma['날짜'] = pd.to_datetime(df_inv_ma['날짜'])
            ma_cols_to_process = [c for c in df_inv_ma.columns if '이동평균' in c]
            for col in ma_cols_to_process:
                temp_pivot = df_inv_ma.pivot_table(index='날짜', columns=['도메인', '지수종류'], values=col).reset_index()
                temp_pivot.columns = ['_'.join(c).strip() if isinstance(c, tuple) else c for c in temp_pivot.columns]
                temp_pivot.columns = [c.replace(f"_{col}", "") + f"_{col}" if c != '날짜' else c for c in
                                      temp_pivot.columns]
                for pivoted_col in [c for c in temp_pivot.columns if c != '날짜']:
                    aligner.add_preprocessed_df(temp_pivot, pivoted_col, '날짜', pivoted_col)

        final_output_path = 'outputs/aligned_final_dataset_all_12_no_nulls.csv'
        final_df = aligner.process_and_save(final_output_path)

        if final_df is not None:
            print("\n모든 데이터가 성공적으로 통합되었습니다.")
            null_counts = final_df.isnull().sum().sum()

            # Software_Investment 컬럼 특별 확인
            if 'Software_Investment' in final_df.columns:
                software_nulls = final_df['Software_Investment'].isnull().sum()
                software_valid = (~final_df['Software_Investment'].isnull()).sum()
                print(f"Software_Investment 컬럼: null={software_nulls}, 유효값={software_valid}")
                print(f"Software_Investment 샘플값: {final_df['Software_Investment'].dropna().head()}")
            else:
                print("Software_Investment 컬럼이 최종 데이터에 없습니다.")

            if null_counts == 0:
                print(">>> 최종 데이터셋에 Null 값이 없습니다. (성공)")
            else:
                print(f">>> 경고: 최종 데이터셋에 {null_counts}개의 Null 값이 남아있습니다.")
                print("컬럼별 null 개수:")
                print(final_df.isnull().sum())

            print("\n최종 통합 데이터 (상위 5개):")
            print(final_df.head())

    except FileNotFoundError as e:
        print(f"\n[오류] 파일 없음: '{e.filename}' 파일을 찾을 수 없습니다. DATA_DIRECTORY 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()